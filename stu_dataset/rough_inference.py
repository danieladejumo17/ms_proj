#!/usr/bin/env python3
"""
Fast Fault Monitor for Vision-Based Autonomous Vehicle
-------------------------------------------------------
Optimized version of the original Cosmos-Reason fault monitor script.
"""

import argparse
from pathlib import Path
import torch
import qwen_vl_utils
import transformers
import warnings
import cv2
import os

warnings.filterwarnings("ignore", category=UserWarning)


def load_model(model_name: str):
    """Loads model in bfloat16 precision."""
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    return model, processor


def main():
    parser = argparse.ArgumentParser(description="Fast Fault Monitor for Self-Driving Car")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, default="nvidia/Cosmos-Reason1-7B", help="Model name or path")
    parser.add_argument("--fps", type=int, default=4, help="Video FPS sampling rate (default: 4)")
    parser.add_argument("--max_tokens", type=int, default=8, help="Max tokens to generate (default: 8)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if args.verbose:
        print(f"Model: {args.model}")
        print(f"Video: {video_path}")
        print(f"FPS: {args.fps}")

    # Load model
    model, processor = load_model(args.model)

    # Prepare full video input (no fast mode)
    content = [
        {
            "type": "video",
            "video": str(video_path),
            "fps": args.fps,
            "total_pixels": 4096 * 30**2,
        }
    ]

    # Updated anomaly detection prompt
    content.append({
        "type": "text",
        "text": (
            "You are an autonomous driving safety expert analyzing this video for EXTERNAL ANOMALIES that may impact safe AV operation.\n\n"
            "<think>\n"
            "Focus on:\n"
            "- Obstacles, pedestrians, or vehicles violating rules\n"
            "- Roadwork, blocked lanes, poor visibility, or hazards\n"
            "- Reflections, shadows, or false visual cues confusing perception\n"
            "</think>\n\n"
            "<answer>\n"
            "Is there any external anomaly in this video? Reply with exactly one of the following:\n"
            "Classification: Anomaly — if any obstacle, obstruction, or unsafe condition is visible.\n"
            "Classification: Normal — if no anomaly or obstruction is visible.\n"
            "</answer>"
        ),
    })

    conversation = [{"role": "user", "content": content}]

    # Prepare model inputs
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(conversation)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    if args.verbose:
        print("Prompt text:\n", text)

    # Inference
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        do_sample=False,
        num_beams=1,
    )

    output_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    output_text = processor.batch_decode(output_ids, skip_special_tokens=True)
    result = output_text[0].strip()

    print(result)


if _name_ == "_main_":
    main()