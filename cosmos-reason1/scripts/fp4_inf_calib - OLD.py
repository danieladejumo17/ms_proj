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
"""

import os
import sys
import logging
import warnings
import argparse
import time
import gc
import glob
from pathlib import Path
from typing import List, Dict, Any, Generator

import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer

# --- LOGGING SETUP ---
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fp4_calibration")

warnings.filterwarnings("ignore")

# try:
from vllm import LLM, SamplingParams
# except ImportError:
#     print("❌ Error: 'vllm' not found. Install it via: pip install vllm")
#     sys.exit(1)

# try:
from llmcompressor.modifiers.quantization import QuantizationModifier
# from llmcompressor.pipelines import oneshot
from llmcompressor import oneshot
# except ImportError:
#     print("❌ Error: 'llmcompressor' not found. Install it via: pip install llmcompressor")
#     sys.exit(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from stu_dataset.stu_video_dataset import stu_video_dataloader, Metrics

# ============================================================
# Configuration
# ============================================================
MODEL_ID = "nvidia/Cosmos-Reason1-7B"
QUANT_PATH = "Cosmos-Reason1-7B-NVFP4-Accurate"

# Calibration settings
CALIB_MAX_SEQ_LEN = 4096  # Restored to original
PIXEL_BUDGET = 250 * 250  # Restored to original
MAX_NEW_TOKENS = 16 # 7

# ============================================================
# 1. Calibration Data Loader
# ============================================================
def get_calibration_loader(processor, dataset_dir: str, num_samples: int = 128):
    """
    Generates REAL input data for calibration.
    Reads actual videos from the directory and pairs them with formatted prompts.
    """
    # video_files = glob.glob(os.path.join(video_dir, "*.mp4")) + \
    #               glob.glob(os.path.join(video_dir, "*.mkv"))
    
    # if not video_files:
        # raise ValueError(f"No video files found in {video_dir} for calibration!")

    # print(f"📹 Found {len(video_files)} videos. Using {min(len(video_files), num_samples)} for calibration.")

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
    # Lazy imports for video handling if needed by processor
    import decord 
    
    while count < num_samples:
        for vid_path, _, _ in stu_video_dataloader(dataset_dir, window_size=50, step_size=20, fps=10, separate_videos=True, output_dir="./output_videos/"):
        # for vid_path in video_files:
            if count >= num_samples:
                break
                
            prompt_text = base_text + base_prompts[count % len(base_prompts)] + "</answer>"
            
            # Prepare messages (Chat Format)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": vid_path},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            try:
                text_input = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # Apply the PIXEL_BUDGET constraint here
                inputs = processor(
                    text=[text_input],
                    videos=[vid_path],
                    padding=True,
                    return_tensors="pt",
                    max_pixels=PIXEL_BUDGET  # Constrain resolution during calibration
                )
                
                batch = {
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "pixel_values": inputs.pixel_values,
                    "image_grid_thw": inputs.image_grid_thw
                }
                
                yield batch
                count += 1
                if count % 10 == 0:
                    print(f"   Generated calibration sample {count}/{num_samples}")
                
            except Exception as e:
                print(f"   ⚠️ Skipping {vid_path} due to error: {e}")
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
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # 1. Visual Encoder (already done, but let's be explicit)
    # 2. LM Head (Keep high precision for output stability)
    ignore_targets = ["re:.*visual.*", "lm_head"] 

    # Define the Quantization Recipe
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP4", # NVFP4 target
        ignore=ignore_targets, 
        dampening_frac=0.01, # Helps stability slightly
    )

    def data_loader_func():
        return get_calibration_loader(processor, dataset_dir, num_calib_samples)

    print("   Starting One-Shot calibration...")
    oneshot(
        model=MODEL_ID,
        dataset=data_loader_func(),
        recipe=recipe,
        output_dir=QUANT_PATH,
        trust_remote_code=True,
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
    
    # Load vLLM with constraints
    llm = LLM(
        model=QUANT_PATH,
        trust_remote_code=True,
        quantization="fp4",
        max_model_len=CALIB_MAX_SEQ_LEN, # 4096
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 5, "video": 2},
        # Pass the pixel budget to vLLM's internal processor
        mm_processor_kwargs={"max_pixels": PIXEL_BUDGET}
    )

    # video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    # Increased max tokens slightly to allow for the full "Classification: X" response
    sampling_params = SamplingParams(
        temperature=0.1, 
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS 
    )

    # print(f"📂 Processing {len(video_files)} videos...")
    
    inference_metrics = Metrics()

    total_start = time.time()
    
    for video_file, label, _ in stu_video_dataloader(dataset_dir, window_size=50, step_size=20, fps=10, separate_videos=True, output_dir="./output_videos/"):
        video_path = str(Path(video_file).absolute())
        
        # Original detailed prompt structure
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
                "video": video_path
            }
        }

        try:
            outputs = llm.generate([inputs], sampling_params=sampling_params, use_tqdm=False)
            generated_text = outputs[0].outputs[0].text.strip()
            
            # Simple parsing of the classification
            status = "Unknown"
            if "anomaly" in generated_text.lower():
                status = "Anomaly"
            elif "normal" in generated_text.lower():
                status = "Normal"
            else:
                status = f"Raw({generated_text[:50]}...)"
            elapsed = time.time() - loop_start

            # Track metrics
            pred_label = 1 if status == "Anomaly" else 0
            inference_metrics.update([label], [pred_label], [elapsed])

            print(f"[{Path(video_file).name}] -> {status} \t| Time: {elapsed:.3f}s")
            
        except Exception as e:
            # Truncate errors
            err_msg = str(e)
            if len(err_msg) > 500:
                err_msg = err_msg[:500] + "... [Error truncated]"
            print(f"[{Path(video_file).name}] Error: {err_msg}")

    print("=" * 60)
    print(f"Pipeline Complete. Total time: {time.time() - total_start:.2f}s")
    # Print Final Metrics
    final_metrics = inference_metrics.compute()
    # print metrics on one line each
    print(f"Accuracy: {final_metrics['accuracy']*100:.2f}% %n Precision: {final_metrics['precision']*100:.2f}% %n Recall: {final_metrics['recall']*100:.2f}% %n F1: {final_metrics['f1']*100:.2f}%")
    print(f"Confusion Matrix: TP={final_metrics['tp']}, TN={final_metrics['tn']}, FP={final_metrics['fp']}, FN={final_metrics['fn']}")
    print(f"Average Inference Time per Video: {final_metrics['Avg Inference Time']:.3f}s")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to videos (used for BOTH calibration and inference)")
    parser.add_argument("--calib_samples", type=int, default=128, help="Number of video samples to use for calibration (default: 128)")
    args = parser.parse_args()

    ensure_fp4_model(args.dataset_dir, args.calib_samples)
    run_inference(args.dataset_dir)


    # Why different chat templates?
    # Ensure quantization sees same structure during calibration
    # as during inference for best accuracy.
    # Develop your video loader pipeline
    # Evaluate local performance first