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