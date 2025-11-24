#!/usr/bin/env python3
"""Batch inference over nuscenes videos.

This script walks /home/daniel/Dev/ms_proj/nuscenes/videos for video files,
runs the same prompt as in `inference_sample.py` with fps=6 and
total_pixels=(4*6*800*450), logs runtime per video to the terminal and
appends the video filename and model output to `batch_inference_results.txt`.

Usage:
    python scripts/batch_inference_nuscenes.py
"""

from pathlib import Path
import time
import sys

import transformers
import qwen_vl_utils


ROOT = Path(__file__).parents[1]
VIDEOS_DIR = Path("/home/daniel/Dev/ms_proj/nuscenes/videos")
OUTPUT_FILE = ROOT / "batch_inference_results_old_paper_prompt_verbose.txt"
SEPARATOR = "-" * 80


def make_conversation(video_path: str):
    # Reuse the same prompt as in inference_sample.py but with the requested fps and total_pixels
    total_pixels = 4 * 6 * 800 * 450
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "fps": 6,
                    "total_pixels": total_pixels,
                },
                { 
                    "type": "text",
                    "text": f"""I am the fault monitor for a vision-based autonomous vehicle. My job is to analyze the vehicle’s observations and identify anything that could cause the vehicle to take actions that are unsafe, unpredictable or violate traffic rules. For each object that the vehicle observes, I will reason about whether the object constitutes a normal observation or an anomaly. Normal observations do not detrimentally affect the vehicle’s performance, whereas anomalies might. Finally, I will classify whether the overall scene is normal or abnormal. For example,

"
The vehicle is driving on the road and
observes:
-a cyclist on the sidewalk
-a car on the road
-a pedestrian carrying a bright green balloon

Cyclist on the sidewalk:
1. Is this common to see while driving? Yes, cyclists can often be seen riding on the road or occasionally on sidewalks.
2. Can this influence the vehicle’s behavior? No, they are on the sidewalk and not on the road.
3. Can the vehicle drive safely in its presence? Yes, cyclists are commonly seen on the road and the vehicle should be able to drive safely in their presence.
4. Can this cause the vehicle to make unpredictable or unsafe maneuvers? No, they are on the sidewalk and not on the road.
Classification: Normal.

Car on the road:
1. Is this common to see while driving? Yes, cars are common to see while driving.
2. Can this influence the vehicle’s behavior? Yes, the autonomous vehicle must respect other vehicles on the road, avoid collisions and obey the rules of the road.
3. Can the vehicle drive safely in its presence? Yes, cars are commonly seen on the road and the autonomous vehicle should be able to drive safely in their presence.
4. Can this cause the vehicle to make unpredictable or unsafe maneuvers? No, autonomous vehicles are programmed to appropriately drive and interact with other cars on the road.
Classification: Normal.

Pedestrian carrying a bright green balloon:
1. Is this common to see while driving? Pedestrians are commonly seen on the sidewalk or crossing at intersections. They may possess arbitrary objects and a balloon is a reasonable object to carry.
2. Can this influence the vehicle’s behavior? Yes, the autonomous vehicle may mistake the green balloon for a green traffic light signal, which could deceive it into driving forward when it should otherwise be stopped (e.g., at a red light).
3. Can the vehicle drive safely in its presence? No, this could deceive the vehicle into interpreting the green balloon as a legal traffic signal.
4. Can this cause the vehicle to make unpredictable or unsafe maneuvers? Yes, this could deceive the autonomous vehicle into driving forward when it should otherwise be stopped (e.g., at a red light)
Classification: Anomaly.

Overall Scenario Classification: Anomaly.
"

<think>
I am driving on the road and I see:
</think>

<answer>
Overall Scenario Classification:
</answer>"""
                },
            ],
        }
    ]
    return conversation


def run_inference_on_video(model, processor, video_path: Path):
    conversation = make_conversation(video_path)  # just modify dict here
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(conversation)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    start = time.perf_counter()
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    elapsed = time.perf_counter() - start

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    # INSPECT GENNERATED IDS
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0] if output_text else "", elapsed, len(generated_ids_trimmed[0])


def main():
    # Load model once
    model_name = "nvidia/Cosmos-Reason1-7B"
    print(f"Loading model {model_name} ...")
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor: transformers.Qwen2_5_VLProcessor = (
        transformers.AutoProcessor.from_pretrained(model_name)
    )

    # Prepare output file
    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")
    out_f.write(f"\n{SEPARATOR}\nBatch run at {time.ctime()}\n{SEPARATOR}\n")

    if not VIDEOS_DIR.exists():
        print(f"Videos directory not found: {VIDEOS_DIR}")
        sys.exit(1)

    # Iterate video files (common video extensions)
    exts = {".mp4", ".mov", ".avi", ".mkv"}
    videos = sorted([p for p in VIDEOS_DIR.iterdir() if p.suffix.lower() in exts])
    if not videos:
        print(f"No videos found in {VIDEOS_DIR}")
        out_f.close()
        return

    for vid in videos:
        print(SEPARATOR)
        print(f"Processing: {vid.name}")
        out_f = open(OUTPUT_FILE, "a", encoding="utf-8")
        out_f.write(f"\nProcessing: {vid.name}\n")

        try:
            output, elapsed, gen_ids_len = run_inference_on_video(model, processor, vid)
        except Exception as e:
            print(f"Error processing {vid.name}: {e}")
            out_f.write(f"Error processing {vid.name}: {e}\n")
            continue

        # Log to terminal
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Num of generated ids: {gen_ids_len}")
        print(output)

        # Append to output file with separators
        out_f.write(f"Time elapsed: {elapsed:.2f}s\n")
        out_f.write(f"Num of generated ids: {gen_ids_len}\n")
        out_f.write(output + "\n")
        out_f.write(SEPARATOR + "\n")
        out_f.close()

    # out_f.close()
    print("Batch inference finished. Results appended to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
