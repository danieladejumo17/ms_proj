#!/usr/bin/env python3
"""
Cosmos-Reason1-7B FP4 Inference Pipeline for RTX 5090
-----------------------------------------------------
1. Auto-downloads 'nvidia/Cosmos-Reason1-7B'.
2. Generates a hardware-accelerated NVFP4 checkpoint (saved locally).
3. Runs high-performance inference using vLLM on video files.

Parameters:
- Model: nvidia/Cosmos-Reason1-7B
- Quantization: NVFP4 (Weights & Activations)
- FPS: 4
- Max Tokens: 7
- Resolution: 250x250 (Pixel budget constrained)
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer

# Import quantization tools
try:
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.transformers import SparseAutoModelForCausalLM, SparseAutoTokenizer
    from llmcompressor import oneshot
except ImportError:
    print("❌ Error: 'llmcompressor' not found. Install it via: pip install llmcompressor")
    sys.exit(1)

# Import vLLM
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("❌ Error: 'vllm' not found. Install it via: pip install vllm")
    sys.exit(1)


# ============================================================
# 1. Configuration
# ============================================================
MODEL_ID = "nvidia/Cosmos-Reason1-7B"
QUANTIZED_MODEL_PATH = "./Cosmos-Reason1-7B-NVFP4"

# Inference Parameters (Matched to fp8_test.py)
FPS = 4
MAX_TOKENS = 7
TARGET_RESOLUTION = (250, 250) # WxH

PROMPT_TEXT = (
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
    "</answer>"
)

# ============================================================
# 2. Quantization Engine (FP4 Generation)
# ============================================================
def ensure_fp4_model():
    """
    Checks if the NVFP4 model exists. If not, downloads the base model
    and quantizes it using llmcompressor.
    """
    if os.path.exists(QUANTIZED_MODEL_PATH) and os.path.exists(os.path.join(QUANTIZED_MODEL_PATH, "config.json")):
        print(f"✅ Found existing FP4 model at: {QUANTIZED_MODEL_PATH}")
        return

    print(f"⚠️ FP4 model not found. Starting generation for {MODEL_ID}...")
    print("⏳ This process runs once. It downloads weights and calculates quantization scales.")

    # 1. Load Base Model (BF16)
    print("   - Loading base model...")
    model = SparseAutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        torch_dtype="auto", 
        trust_remote_code=True
    )
    tokenizer = SparseAutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 2. Configure NVFP4 Recipe
    # We target Linear layers in the LLM. The Vision Encoder usually remains BF16 
    # to preserve visual acuity, which llmcompressor handles by default (targets="Linear").
    print("   - Configuring NVFP4 recipe...")
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4", 
        ignore=["lm_head"]  # Keep head precise for stability
    )

    # 3. Apply Quantization (Calibration)
    # For true FP4 accuracy, we usually use a calibration dataset. 
    # To keep this script standalone and fast, we use a minimal synthetic set 
    # or no calibration (data-free) if the model allows. 
    # Here we use a tiny calibration run to initialize scales.
    print("   - Quantizing (this may take a minute)...")
    
    # Create a dummy calibration dataset to set scales
    ds = [{"text": "Autonomous driving analysis requires identifying pedestrians and obstacles."} * 10]
    
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        num_calibration_samples=1,
        max_seq_length=128
    )

    # 4. Save
    print(f"   - Saving to {QUANTIZED_MODEL_PATH}...")
    model.save_pretrained(QUANTIZED_MODEL_PATH, save_compressed=True)
    tokenizer.save_pretrained(QUANTIZED_MODEL_PATH)
    print("✅ FP4 Model Generated successfully.\n")

# ============================================================
# 3. Video Processing (CPU Prefetch Logic)
# ============================================================
def process_video_frames(video_path: str, fps: int, resolution: tuple) -> List[np.ndarray]:
    """
    Decodes video, samples FPS, and resizes to target resolution.
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0: native_fps = 24.0
    
    frame_interval = max(1, int(round(native_fps / fps)))
    width, height = resolution

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            # Resize to target resolution (250x250)
            frame = cv2.resize(frame, (width, height))
            # Convert BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        count += 1
    
    cap.release()
    return frames

# ============================================================
# 4. Inference Logic
# ============================================================
def run_inference_loop(video_dir: str):
    # 1. Initialize vLLM with the Quantized Model
    print("🚀 Initializing vLLM engine with FP4 support...")
    
    # Cosmos-Reason1 is based on Qwen2.5-VL. vLLM supports this.
    # We assume the quantization config in the saved model is correctly 'modelopt' or 'fp8' format.
    # If llmcompressor saved as NVFP4, vLLM loads it via quantization="modelopt".
    try:
        llm = LLM(
            model=QUANTIZED_MODEL_PATH,
            quantization="modelopt", # 'modelopt' is the loader for NVFP4 checkpoints
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=False, # CUDA Graphs on for speed
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"video": 1}
        )
    except Exception as e:
        print(f"vLLM Init Error: {e}")
        print("Fallback: Trying quantization='fp8' if hardware requires...")
        llm = LLM(model=QUANTIZED_MODEL_PATH, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(QUANTIZED_MODEL_PATH, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    # 2. Find Videos
    video_path_obj = Path(video_dir)
    video_files = sorted([f for ext in ("*.mp4", "*.avi", "*.mov") for f in video_path_obj.glob(ext)])
    
    if not video_files:
        print("No videos found.")
        return

    print(f"\nStarting Inference on {len(video_files)} videos...")
    print(f"Parameters: FPS={FPS}, Res={TARGET_RESOLUTION}, MaxTokens={MAX_TOKENS}")
    print("=" * 60)

    total_start = time.time()

    for video_file in video_files:
        # A. Prepare Inputs
        # vLLM for VLMs accepts video inputs directly if formatted correctly, 
        # but often passing raw pixel values is safer for custom pipelines.
        # Here we pass the raw frames we extracted.
        frames = process_video_frames(video_file, FPS, TARGET_RESOLUTION)
        
        if not frames:
            continue

        # B. Construct Prompt
        # Qwen2.5/Cosmos template construction
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": frames}, # vLLM handles the NumPy array list
                {"type": "text", "text": PROMPT_TEXT}
            ]
        }]

        # Apply template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # C. Inference
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"video": frames},
        }

        loop_start = time.time()
        try:
            outputs = llm.generate([inputs], sampling_params=sampling_params, use_tqdm=False)
            generated_text = outputs[0].outputs[0].text.strip()
            
            # Parse result
            status = "Unknown"
            if "anomaly" in generated_text.lower():
                status = "Anomaly"
            elif "normal" in generated_text.lower():
                status = "Normal"
            else:
                status = f"Raw({generated_text})"

            elapsed = time.time() - loop_start
            print(f"[{video_file.name}] -> {status} \t| Time: {elapsed:.3f}s")
            
        except Exception as e:
            print(f"[{video_file.name}] Error: {e}")

    print("=" * 60)
    print(f"Pipeline Complete. Total time: {time.time() - total_start:.2f}s")

# ============================================================
# 5. Main Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True, help="Path to directory containing videos")
    args = parser.parse_args()

    # Step 1: Ensure Model Exists
    ensure_fp4_model()

    # Step 2: Run Inference
    run_inference_loop(args.video_dir)