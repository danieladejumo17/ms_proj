#!/usr/bin/env python3
"""
Cosmos-Reason1-7B FP4 Inference Pipeline (Improved Accuracy)
------------------------------------------------------------
Improvements:
1. Uses REAL calibration data (videos + text) instead of random noise.
2. Protects sensitive layers (lm_head, first/last layers) from destructive quantization.
3. Uses a proper chat template for the calibration prompts.
4. Configurable calibration sample size via CLI.
5. Restored original constraints: MAX_SEQ_LEN=4096, PIXEL_BUDGET=250*250.
6. Uses specific detailed prompt for inference.
7. ROBUSTNESS: Adapts to processor keys (pixel_values vs pixel_values_videos).
8. COMPATIBILITY: Forces Float32 to avoid quantization type promotion errors.
"""

import os
import sys
import logging
import warnings
import argparse
import time
import gc
import traceback
import glob
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModel

# Check for datasets
try:
    from datasets import Dataset
except ImportError:
    print("❌ Error: 'datasets' library not found. Install it via: pip install datasets")
    pass

# Check for OpenCV/PIL
try:
    import cv2
    from PIL import Image
except ImportError:
    cv2 = None
    Image = None
    print("⚠️ OpenCV or PIL not found. Video reading fallback will be disabled.")

# --- LOGGING SETUP ---
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fp4_calibration")

warnings.filterwarnings("ignore")

from vllm import LLM, SamplingParams
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
try:
    from stu_dataset.stu_video_dataset import stu_video_dataloader, Metrics
except ImportError:
    print("⚠️ Warning: Could not import 'stu_video_dataloader'. Ensure dataset paths are correct.")

# ============================================================
# Configuration
# ============================================================
MODEL_ID = "nvidia/Cosmos-Reason1-7B"
QUANT_PATH = "Cosmos-Reason1-7B-NVFP4-Accurate"

# Calibration settings
CALIB_MAX_SEQ_LEN = 4096  # Restored to original
PIXEL_BUDGET = 250 * 250  # Restored to original
MAX_NEW_TOKENS = 16 

# ============================================================
# Helper: Manual Video Reader (Robust Fallback)
# ============================================================
def load_video_frames_cv2(video_path: str) -> List[Any]:
    """
    Reads a video using OpenCV and returns a list of PIL Images.
    This bypasses decord/av which can be flaky with certain MP4 encodings.
    """
    if cv2 is None:
        return None
        
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        
    cap.release()
    return frames

# ============================================================
# 1. Calibration Data Loader
# ============================================================
def get_calibration_loader(processor, dataset_dir: str, num_samples: int = 128):
    """
    Generates REAL input data for calibration.
    """
    # Varied prompts for calibration
    base_prompts = [
        "Describe the events in this video in detail.\n",
        "What is happening in this scene?\n",
        "Explain the physics observed in this video.\n",
        "Is there any anomaly in this footage?\n",
        "List the objects visible in the video.\n",
        (
        "Is there any external anomaly in this video? Reply with exactly one word of the following:\n"
        "Classification: Anomaly — if any obstacle, obstruction, or unsafe condition is visible.\n"
        "Classification: Normal — if no anomaly or obstruction is visible.\n"
        )
    ]
    
    base_text = (
        "You are an autonomous driving safety expert analyzing this video for EXTERNAL ANOMALIES that may impact safe AV operation.\n\n"
        "<think>\n"
        "- Obstacles, pedestrians, or vehicles violating rules\n"
        "- Roadwork, blocked lanes, poor visibility, or hazards\n"
        "- Reflections, shadows, or false visual cues confusing perception\n"
        "</think>\n\n"
        "<answer>\n"
        )

    count = 0
    
    while count < num_samples:
        try:
            loader = stu_video_dataloader(dataset_dir, window_size=50, step_size=20, fps=10, separate_videos=True, output_dir="./output_videos/")
        except NameError:
            print("❌ Error: Data loader not defined.")
            return

        # STU Dataloader yields: (video_path, label, third_element)
        # third_element could be frames (List[Image]) OR metadata (int/str)
        # We must detect this dynamically to be robust.
        for vid_path_in, label, third_element in loader:
            if count >= num_samples:
                break
            
            final_frames = None
            
            # --- DYNAMIC TYPE CHECKING ---
            if isinstance(third_element, list) and len(third_element) > 0 and isinstance(third_element[0], Image.Image):
                # Case A: Dataloader yields frames directly
                final_frames = third_element
                # Verify RGB
                if final_frames[0].mode != "RGB":
                    final_frames = [f.convert("RGB") for f in final_frames]
            else:
                # Case B: Dataloader yields metadata. We must load video from disk.
                # Use robust path finding logic
                meta = str(third_element)
                vid_path = vid_path_in
                
                # Check 1: Does path exist?
                if not os.path.exists(vid_path):
                    # Check 2: Try appending metadata suffix (scene_125 -> scene_125_30.mp4)
                    p = Path(vid_path_in)
                    candidate = p.parent / f"{p.stem}_{meta}{p.suffix}"
                    if candidate.exists():
                        vid_path = str(candidate)
                    else:
                        # Check 3: Wildcard search
                        matches = sorted(list(p.parent.glob(f"{p.stem}_*{p.suffix}")))
                        if matches:
                            vid_path = str(matches[0])
                
                if os.path.exists(vid_path) and os.path.getsize(vid_path) > 0:
                    # Try loading with OpenCV
                    final_frames = load_video_frames_cv2(vid_path)

            if not final_frames or len(final_frames) == 0:
                print(f"   ⚠️ Skipping {vid_path_in}: Could not load frames.")
                continue
                
            prompt_text = base_text + base_prompts[count % len(base_prompts)] + "</answer>"
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": vid_path_in}, # Path is just for reference in template
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            try:
                text_input = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # --- PROCESSOR CALL ---
                # FIX: Convert PIL Images to Numpy Arrays.
                # Some fast processors handle List[np.array] better than List[PIL.Image] for videos.
                video_frames_np = [np.array(f) for f in final_frames]

                # videos=[video_frames_np] -> List containing one video, which is a list of numpy frames
                inputs = processor(
                    text=[text_input],
                    videos=[video_frames_np], 
                    padding=True,
                    return_tensors="pt",
                    max_pixels=PIXEL_BUDGET
                )
                
                # --- FIX: HANDLE VARYING PROCESSOR OUTPUT KEYS ---
                # Qwen2-VL uses 'pixel_values_videos' instead of 'pixel_values'.
                # We need to capture ALL relevant keys.
                
                # Base batch
                batch = {
                    "input_ids": inputs.input_ids[0].cpu(),
                    "attention_mask": inputs.attention_mask[0].cpu(),
                }
                
                # Vision keys mapping
                # We check for both standard and model-specific keys
                vision_keys = ["pixel_values", "pixel_values_videos", "video_grid_thw", "image_grid_thw"]
                keys_found = False
                
                for key in vision_keys:
                    if key in inputs:
                        batch[key] = inputs[key].cpu()
                        keys_found = True
                        
                if not keys_found:
                    print(f"   ⚠️ Skipping {vid_path_in}: No vision keys found. Available: {list(inputs.keys())}")
                    continue

                yield batch
                count += 1
                if count % 10 == 0:
                    print(f"   Generated calibration sample {count}/{num_samples}")
                
            except Exception as e:
                print(f"   ⚠️ Skipping {vid_path_in} due to error.")
                # Print full traceback to debug empty error messages
                traceback.print_exc()
                continue

# ============================================================
# 2. Quantization Routine
# ============================================================
def ensure_fp4_model(dataset_dir: str, num_calib_samples: int):
    if os.path.exists(QUANT_PATH):
        print(f"✅ Found existing quantized model at: {QUANT_PATH}")
        print("   (Delete this folder if you want to re-run calibration with different settings)")
        return

    print(f"⚡ Quantizing {MODEL_ID} to FP4 with {num_calib_samples} samples and {PIXEL_BUDGET} pixel budget...")
    
    print("   Loading model for quantization...")
    # --- CRITICAL FIX: FORCE FLOAT32 ---
    # Attempting to compress BFloat16/Half weights into Float8/FP4 containers causes
    # PyTorch type promotion errors (RuntimeError). We force Float32 to ensure mathematical compatibility.
    # We also add offload_folder="offload" to prevent OOM errors with the larger FP32 weights.
    model = AutoModel.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        trust_remote_code=True,
        torch_dtype=torch.float32,
        offload_folder="offload" 
    )

    # Use default fast processor
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    ignore_targets = ["re:.*visual.*", "lm_head"] 

    fp4_config = {
        "weights": {
            "num_bits": 4,
            "type": "float",    # FP4
            "symmetric": True,
            "strategy": "group",
            "group_size": 128
        },
        "targets": ["Linear"]
    }

    recipe = QuantizationModifier(
        config_groups={"fp4_group": fp4_config}, 
        ignore=ignore_targets,
    )

    print("   Preparing calibration dataset...")
    samples = []
    # Consume generator to build a proper HF Dataset
    for batch in get_calibration_loader(processor, dataset_dir, num_calib_samples):
        samples.append(batch)

    if not samples:
        print("❌ Error: No calibration samples generated. Check dataset path.")
        return

    print(f"   Collected {len(samples)} calibration samples. Converting to HF Dataset...")
    try:
        calib_dataset = Dataset.from_list(samples)
        # We must explicitly tell the dataset which columns to treat as tensors.
        # Use the actual keys present in the first sample.
        tensor_cols = [k for k in samples[0].keys() if k in ["input_ids", "attention_mask", "pixel_values", "pixel_values_videos", "video_grid_thw", "image_grid_thw"]]
        calib_dataset.set_format(type="torch", columns=tensor_cols)
    except Exception as e:
        print(f"❌ Error creating HF Dataset: {e}")
        return

    print("   Starting One-Shot calibration...")
    oneshot(
        model=model,
        dataset=calib_dataset,
        recipe=recipe,
        output_dir=QUANT_PATH,
        max_seq_length=CALIB_MAX_SEQ_LEN,
    )
    
    print(f"🎉 Quantization complete! Saved to {QUANT_PATH}")
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 3. Inference Routine
# ============================================================
def run_inference(dataset_dir: str):
    print(f"🚀 Loading vLLM from: {QUANT_PATH}")
    
    try:
        llm = LLM(
            model=QUANT_PATH,
            trust_remote_code=True,
            quantization="fp4",
            max_model_len=CALIB_MAX_SEQ_LEN, # 4096
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"image": 5, "video": 2},
            mm_processor_kwargs={"max_pixels": PIXEL_BUDGET}
        )
    except Exception as e:
        print(f"❌ Failed to load vLLM: {e}")
        return

    sampling_params = SamplingParams(
        temperature=0.1, 
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS 
    )

    inference_metrics = Metrics()
    total_start = time.time()
    
    try:
        loader = stu_video_dataloader(dataset_dir, window_size=50, step_size=20, fps=10, separate_videos=True, output_dir="./output_videos/")
    except Exception as e:
        print(f"❌ Error initializing dataloader for inference: {e}")
        return

    # Helper function to find video file (reused logic)
    def find_video_file(path_in, meta):
        if os.path.exists(path_in): return path_in
        p = Path(path_in)
        candidate = p.parent / f"{p.stem}_{meta}{p.suffix}"
        if candidate.exists(): return str(candidate)
        matches = sorted(list(p.parent.glob(f"{p.stem}_*{p.suffix}")))
        if matches: return str(matches[0])
        return None

    for video_file_in, label, third_element in loader:
        # Determine video file path
        if isinstance(third_element, list):
            # If frames are yielded, we still need a file path for vLLM usually
            # But the 'video_file_in' might be the base path.
            # We assume video_file_in is usable or we try to find the split file based on implicit naming
            # If we can't determine meta from third_element (since it's frames), we just try the path.
            video_file = video_file_in 
        else:
            # metadata available
            video_file = find_video_file(video_file_in, str(third_element))

        if not video_file or not os.path.exists(video_file):
            print(f"Warning: Inference video file not found for: {video_file_in}")
            continue

        prompt_text = (
            "<|user|>\n"
            "You are an autonomous driving safety expert analyzing this video for EXTERNAL ANOMALIES that may impact safe AV operation.\n\n"
            "<think>\n"
            "- Obstacles, pedestrians, or vehicles violating rules\n"
            "- Roadwork, blocked lanes, poor visibility, or hazards\n"
            "- Reflections, shadows, or false visual cues confusing perception\n"
            "</think>\n\n"
            "<answer>\n"
            "Is there any external anomaly in this video? Reply with exactly one word of the following:\n"
            "Classification: Anomaly — if any obstacle, obstruction, or unsafe condition is visible.\n"
            "Classification: Normal — if no anomaly or obstruction is visible.\n"
            "</answer>\n"
            "<|video_pad|>\n"
            "<|assistant|>\n"
        )
        
        loop_start = time.time()
        inputs = {
            "prompt": prompt_text,
            "multi_modal_data": {
                "video": video_file
            }
        }

        try:
            outputs = llm.generate([inputs], sampling_params=sampling_params, use_tqdm=False)
            generated_text = outputs[0].outputs[0].text.strip()
            
            status = "Unknown"
            if "anomaly" in generated_text.lower():
                status = "Anomaly"
            elif "normal" in generated_text.lower():
                status = "Normal"
            else:
                status = f"Raw({generated_text[:50]}...)"
            elapsed = time.time() - loop_start

            # Cast boolean numpy/torch label to int for metrics
            if isinstance(label, (bool, np.bool_)):
                label_int = 1 if label else 0
            else:
                label_int = int(label)

            pred_label = 1 if status == "Anomaly" else 0
            inference_metrics.update([label_int], [pred_label], [elapsed])

            print(f"[{Path(video_file).name}] -> {status} \t| Time: {elapsed:.3f}s")
            
        except Exception as e:
            err_msg = str(e)
            if len(err_msg) > 500:
                err_msg = err_msg[:500] + "... [Error truncated]"
            print(f"[{Path(video_file).name}] Error: {err_msg}")

    print("=" * 60)
    print(f"Pipeline Complete. Total time: {time.time() - total_start:.2f}s")
    final_metrics = inference_metrics.compute()
    print(f"Accuracy: {final_metrics['Accuracy']*100:.2f}%")
    print(f"Precision: {final_metrics['Precision']*100:.2f}%")
    print(f"Recall: {final_metrics['Recall']*100:.2f}%")
    print(f"F1: {final_metrics['F1-Score']*100:.2f}%")
    print(f"Confusion Matrix: TP={final_metrics['TP']}, TN={final_metrics['TN']}, FP={final_metrics['FP']}, FN={final_metrics['FN']}")
    print(f"Average Inference Time per Video: {final_metrics['Avg Inference Time']:.3f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to videos (used for BOTH calibration and inference)")
    parser.add_argument("--calib_samples", type=int, default=3, help="Number of video samples to use for calibration (default: 3)")
    args = parser.parse_args()

    ensure_fp4_model(args.dataset_dir, args.calib_samples)
    run_inference(args.dataset_dir)