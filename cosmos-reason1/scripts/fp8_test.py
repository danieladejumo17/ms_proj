#!/usr/bin/env python3
"""
Fast Fault Monitor (Hybrid Prefetch Version)
--------------------------------------------
Optimized for single-GPU efficiency:
- Loads Qwen2.5-VL once (8-bit + FlashAttention2)
- Performs torch.compile warmup for speed
- Sequential GPU inference (no contention)
- Prefetches next video on CPU for overlap
- Cached static text prompt
"""
import os
os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'

import argparse
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
        # attn_implementation="flash_attention_2",
        device_map="auto",
        # local_files_only=True
    ).eval()

    processor = transformers.AutoProcessor.from_pretrained(model_name) #, local_files_only=True)
    model.gradient_checkpointing_disable()
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)#, mode="reduce-overhead")

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
    # This step is still useful to ensure the text part of the prompt is well-formed
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
# 4. Output Parsing
# ============================================================
def parse_result(raw_output: str) -> str:
    out = raw_output.lower()
    if "anomaly" in out:
        return "Anomaly"
    elif "normal" in out:
        return "Normal"
    return "Unknown"


# ============================================================
# 5. Video Analysis (Corrected)
# ============================================================
def analyze_video(model, processor, video_path: Path, prefetched_data, max_tokens: int, base_text: str):
    # CHANGED: This function now accepts the video_path again, which is needed
    # to correctly generate the text prompt with the <video> placeholder.
    image_inputs, video_inputs = prefetched_data

    # **THE FIX IS HERE**: We construct the full conversation object, including a reference
    # to the video. This tells `apply_chat_template` to insert the <video> token.
    # The video path here isn't used for decoding, only for templating.
    content = [
        {"type": "video", "video": str(video_path)}, 
        {"type": "text", "text": base_text},
    ]
    conversation = [{"role": "user", "content": content}]
    
    # This will now correctly generate a text string containing the <video> token.
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    
    # Now, we pass the text (with placeholder) AND the prefetched video tensors.
    # The processor will match them up correctly.
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    
    new_tokens = output[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


# ============================================================
# 6. Prefetch Next Video (CPU)
# ============================================================
# ============================================================
# 6. Prefetch Next Video (CPU) - OPTIMIZED
# ============================================================
def prefetch_video(video_path: Path, target_fps: int, target_resolution: tuple[int, int]):
    """
    Decode/prepare next video on CPU with a dynamically calculated AND CAPPED pixel budget
    to prevent excessive memory usage with long videos.
    """
    # A safe upper limit to prevent OOM errors.
    # This corresponds to ~128 frames at 384x384 resolution.
    #MAX_PIXEL_BUDGET = 384 * 384 * 128 

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        effective_fps = min(target_fps, native_fps) if native_fps > 0 else target_fps
        duration_seconds = frame_count / native_fps if native_fps > 0 else 0
        num_frames_to_sample = max(1, int(duration_seconds * effective_fps))
        
        # Calculate the theoretical pixel budget
        calculated_pixels = num_frames_to_sample * target_resolution[0] * target_resolution[1]
        
        # **THE FIX IS HERE**: Enforce the maximum budget.
        #total_pixels = min(calculated_pixels, MAX_PIXEL_BUDGET)
        total_pixels = calculated_pixels
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()

    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(
        [{"role": "user", "content": [{"type": "video", "video": str(video_path), "total_pixels": total_pixels}]}]
    )
    return image_inputs, video_inputs

# ============================================================
# 7. Main (With Diagnostic Timers)
# ============================================================
# ============================================================
# 7. Main (With Diagnostic Timers) - CORRECTED
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Fast Fault Monitor - Hybrid Prefetch")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="nvidia/Cosmos-Reason1-7B")
    parser.add_argument("--fps", type=int, default=4, help="Target frames per second to sample from the video.")
    parser.add_argument("--max_tokens", type=int, default=7)
    parser.add_argument("--target_resolution", type=str, default="250x250", help="Target resolution for each frame. This influences the pixel budget.")
    args = parser.parse_args()

    try:
        width, height = map(int, args.target_resolution.split('x'))
        target_resolution = (width, height)
    except ValueError:
        raise ValueError("Invalid format for --target_resolution. Use WxH format, e.g., '224x224'.")

    video_dir = Path(args.video_dir)
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    video_files = sorted([f for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv") for f in video_dir.glob(ext)])
    if not video_files:
        print("No video files found.")
        return

    model, processor = load_model(args.model)
    warmup_model(model, processor)
    base_text = build_cached_prompt(processor)

    print(f"📂 Found {len(video_files)} videos — starting hybrid-prefetch inference\n" + "=" * 50)

    # prefetch_executor = ThreadPoolExecutor(max_workers=1)
    
    # --- THIS IS THE CORRECTED LINE ---
    # next_future = prefetch_executor.submit(prefetch_video, video_files[0], args.fps, target_resolution)
    
    total_start_time = time.time()

    for i, video_path in enumerate(video_files):
        # if not ("scene-0061_CAM_FRONT.mp4" in str(video_path)): # TODO
            # continue
        prefetch_start = time.time()
        # prefetched_data = next_future.result()
        # prefetch_video(video_path: Path, target_fps: int, target_resolution: tuple[int, int]):
        prefetched_data = prefetch_video(video_path, args.fps, target_resolution)
        prefetch_time = time.time() - prefetch_start

        # if i + 1 < len(video_files):
            # next_future = prefetch_executor.submit(prefetch_video, video_files[i + 1], args.fps, target_resolution)

        analysis_start = time.time()
        try:
            # The 'args.fps' was correctly removed from this call in the previous fix
            raw = analyze_video(model, processor, video_path, prefetched_data, args.max_tokens, base_text)
            # print(f"Raw output: {raw}")  # TODO Diagnostic print to see the raw model output
            result = parse_result(raw)
            analysis_time = time.time() - analysis_start
            total_time = time.time() - total_start_time

            print(
                f"[{video_path.name}] -> {result} | "
                f"Total: {total_time:.2f}s "
                f"(Prefetch Wait: {prefetch_time:.2f}s, GPU Analysis: {analysis_time:.2f}s)"
            )
            total_start_time = time.time()

        except Exception as e:
            print(f"[{video_path.name}] ERROR: {e}")

    print("=" * 50 + "\n✅ Hybrid batch processing complete.")


if __name__ == "__main__":
    main()