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
import logging
import warnings

# --- LOGGING & WARNING SUPPRESSION ---
# 1. Silence Library Loggers
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.ERROR)
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# 2. Silence Python Warnings (often contain data dumps)
warnings.filterwarnings("ignore")
# ---------------------------

import argparse
import time
import gc
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# --- CRITICAL IMPORT ORDER FIX ---
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("❌ Error: 'vllm' not found. Install it via: pip install vllm")
    sys.exit(1)

import torch
import cv2
from datasets import Dataset

# 3. Import Transformers & Silence Internal Logger
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor, logging as hf_logging
hf_logging.set_verbosity_error()

try:
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor import oneshot
except ImportError:
    print("❌ Error: 'llmcompressor' not found. Install it via: pip install llmcompressor")
    sys.exit(1)

# ============================================================
# 1. Configuration
# ============================================================
MODEL_ID = "nvidia/Cosmos-Reason1-7B"
QUANT_SAVE_DIR = "./cosmos-reason1-fp4-quant"

# FIX: Increased MAX_SEQ_LEN from 2048 to 4096.
# Video inputs generate thousands of tokens (length was ~3719 in your error).
MAX_SEQ_LEN = 4096 

PIXEL_BUDGET = 250 * 250

# ============================================================
# 2. Quantization Pipeline (One-Shot)
# ============================================================
def ensure_fp4_model():
    """
    Checks for local FP4 model. If missing, downloads base model,
    quantizes to NVFP4 using llmcompressor, and saves.
    """
    has_model = os.path.exists(os.path.join(QUANT_SAVE_DIR, "config.json"))
    has_processor = os.path.exists(os.path.join(QUANT_SAVE_DIR, "preprocessor_config.json"))

    if has_model and has_processor:
        print(f"✅ Found existing FP4 model at: {QUANT_SAVE_DIR}")
        return

    if has_model and not has_processor:
        print(f"⚠️ Found model but missing processor config. Fetching...")
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        if hasattr(processor, "image_processor"):
            processor.image_processor.save_pretrained(QUANT_SAVE_DIR)
        processor.save_pretrained(QUANT_SAVE_DIR)
        print("✅ Processor config saved. Skipping re-quantization.")
        return

    print(f"⚠️ FP4 model not found. Starting quantization of {MODEL_ID}...")
    print("(This process runs once and may take a few minutes)")

    print("   -> Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        torch_dtype="auto", 
        trust_remote_code=True
    )

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=["lm_head", "re:.*visual.*"] 
    )

    print("   -> Generating synthetic calibration data...")
    num_samples = 16
    input_ids_list = [
        np.random.randint(0, 10000, (MAX_SEQ_LEN,)).tolist()
        for _ in range(num_samples)
    ]
    attention_mask_list = [
        np.ones((MAX_SEQ_LEN,), dtype=int).tolist()
        for _ in range(num_samples)
    ]
    ds = Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list
    })

    print("   -> Applying NVFP4 Quantization...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=num_samples,
    )

    print(f"   -> Saving to {QUANT_SAVE_DIR}...")
    model.save_pretrained(QUANT_SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(QUANT_SAVE_DIR)
    
    if hasattr(processor, "image_processor"):
        processor.image_processor.save_pretrained(QUANT_SAVE_DIR)
    processor.save_pretrained(QUANT_SAVE_DIR) 
    
    print("✅ Quantization Complete.")

# ============================================================
# 3. Video Processing Utilities
# ============================================================
def extract_frames(video_path: str, fps: int = 4) -> List[Any]:
    """Extracts frames from video at specified FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(max(1, video_fps / fps))
    
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            # Resize to max 250x250 to fit pixel budget and reduce token count
            frame = cv2.resize(frame, (250, 250))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        count += 1
    cap.release()
    return frames

# ============================================================
# 4. Inference Loop
# ============================================================
def run_inference(video_dir: str):
    video_path = Path(video_dir)
    video_files = list(video_path.glob("*.mp4")) + list(video_path.glob("*.avi"))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return

    print(f"🚀 Loading vLLM engine with model: {QUANT_SAVE_DIR}")
    
    gc.collect()
    torch.cuda.empty_cache()

    llm = LLM(
        model=QUANT_SAVE_DIR, 
        trust_remote_code=True,
        max_model_len=MAX_SEQ_LEN,
        gpu_memory_utilization=0.8,
        enforce_eager=False 
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=128,
        stop=["<|endoftext|>"]
    )

    prompt = (
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

    print(f"Processing {len(video_files)} videos...")
    total_start = time.time()

    for video_file in video_files:
        frames = extract_frames(str(video_file), fps=4)
        
        inputs = {
            "prompt": prompt,
            # Frames wrapped in list to treat as single video item
            "multi_modal_data": {
                "video": [frames] 
            },
        }

        loop_start = time.time()
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
            print(f"[{video_file.name}] -> {status} \t| Time: {elapsed:.3f}s")
            
        except Exception as e:
            # Truncate errors to prevent console flooding
            err_msg = str(e)
            if len(err_msg) > 500:
                err_msg = err_msg[:500] + "... [Error truncated]"
            print(f"[{video_file.name}] Error: {err_msg}")

    print("=" * 60)
    print(f"Pipeline Complete. Total time: {time.time() - total_start:.2f}s")

# ============================================================
# 5. Main Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True, help="Path to directory containing videos")
    args = parser.parse_args()

    ensure_fp4_model()
    run_inference(args.video_dir)


# # Install Dependencies
# pip install vllm transformers llmcompressor datasets opencv-python numpy
# decord