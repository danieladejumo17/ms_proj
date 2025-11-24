#!/usr/bin/env python3
"""
Fast Fault Monitor (Sequential Version — Hybrid Prefetch Disabled)
------------------------------------------------------------------
This version removes hybrid prefetch entirely. Videos are processed
sequentially with no background threads or prefetching.
"""

import argparse
import time
import warnings
from pathlib import Path

import torch
import transformers
import qwen_vl_utils
import cv2

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1. Model Loader
# ============================================================
def load_model(model_name: str):
    print("🔧 Loading and compiling model... This may take a few seconds.")
    start = time.time()

    bnb_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
    ).eval()

    processor = transformers.AutoProcessor.from_pretrained(model_name)
    model.gradient_checkpointing_disable()
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)

    print(f"✅ Model ready in {time.time() - start:.2f}s\n")
    return model, processor


# ============================================================
# 2. Prompt Caching
# ============================================================
def build_cached_prompt(processor):
    base_text = (
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
    conversation_template = [{"role": "user", "content": [{"type": "text", "text": base_text}]}]
    _ = processor.apply_chat_template(conversation_template, tokenize=False, add_generation_prompt=True)
    return base_text


# ============================================================
# 3. Warmup
# ============================================================
def warmup_model(model, processor):
    print("🔥 Warming up model (compiling kernels)...")
    dummy_conv = [{"role": "user", "content": [{"type": "text", "text": "Is this scene safe?"}]}]
    text = processor.apply_chat_template(dummy_conv, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=7)
    torch.cuda.synchronize()
    print("✅ Warmup complete.\n")


# ============================================================
# 4. Result Parsing
# ============================================================
def parse_result(raw_output: str) -> str:
    out = raw_output.lower()
    if "anomaly" in out:
        return "Anomaly"
    elif "normal" in out:
        return "Normal"
    return "Unknown"


# ============================================================
# 5. Video Prefetch (Now Just Decoder — Sequential)
# ============================================================
def load_video(video_path: Path, target_fps: int, target_resolution: tuple[int, int]):
    """
    Sequential (non-prefetch) version of video decoding.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        effective_fps = min(target_fps, native_fps) if native_fps > 0 else target_fps
        duration_seconds = frame_count / native_fps if native_fps > 0 else 0
        num_frames_to_sample = max(1, int(duration_seconds * effective_fps))

        total_pixels = num_frames_to_sample * target_resolution[0] * target_resolution[1]
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()

    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(
        [{"role": "user", "content": [{"type": "video", "video": str(video_path), "total_pixels": total_pixels}]}]
    )
    return image_inputs, video_inputs


# ============================================================
# 6. Video Analysis
# ============================================================
def analyze_video(model, processor, video_path: Path, prefetched_data, max_tokens: int, base_text: str):
    image_inputs, video_inputs = prefetched_data

    content = [
        {"type": "video", "video": str(video_path)},
        {"type": "text", "text": base_text},
    ]
    conversation = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    new_tokens = output[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


# ============================================================
# 7. Main — Sequential Video Processing
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Fast Fault Monitor — Sequential (No Prefetch)")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="nvidia/Cosmos-Reason1-7B")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=7)
    parser.add_argument("--target_resolution", type=str, default="250x250")
    args = parser.parse_args()

    width, height = map(int, args.target_resolution.split('x'))
    target_resolution = (width, height)

    video_dir = Path(args.video_dir)
    video_files = sorted([f for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv") for f in video_dir.glob(ext)])
    if not video_files:
        print("No video files found.")
        return

    model, processor = load_model(args.model)
    warmup_model(model, processor)
    base_text = build_cached_prompt(processor)

    print(f"📂 Found {len(video_files)} videos — running SEQUENTIAL inference (no prefetch)\n" + "=" * 50)

    total_start_time = time.time()

    for video_path in video_files:
        load_start = time.time()
        prefetched_data = load_video(video_path, args.fps, target_resolution)
        load_time = time.time() - load_start

        analysis_start = time.time()
        try:
            raw = analyze_video(model, processor, video_path, prefetched_data, args.max_tokens, base_text)
            result = parse_result(raw)
            analysis_time = time.time() - analysis_start
            total_time = time.time() - total_start_time

            print(
                f"[{video_path.name}] -> {result} | Total: {total_time:.2f}s "
                f"(Load: {load_time:.2f}s, GPU: {analysis_time:.2f}s)"
            )
            total_start_time = time.time()

        except Exception as e:
            print(f"[{video_path.name}] ERROR: {e}")

    print("=" * 50 + "\n✅ Sequential batch processing complete.")


if __name__ == "__main__":
    main()
