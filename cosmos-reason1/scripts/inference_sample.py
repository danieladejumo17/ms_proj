# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal example of inference with Cosmos-Reason1.

Example:

```shell
uv run scripts/inference_sample.py
```
"""

from pathlib import Path

import qwen_vl_utils
import transformers

ROOT = Path(__file__).parents[1]    # ROOT = Path("/home/dxa239/cosmos-reason1")
SEPARATOR = "-" * 20


def main():
    # Load model
    model_name = "nvidia/Cosmos-Reason1-7B"
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor: transformers.Qwen2_5_VLProcessor = (
        transformers.AutoProcessor.from_pretrained(model_name)
    )

    # Create inputs
    conversation = [
        { 
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"{ROOT}/assets/Anom1.mp4",
                    "fps": 30, # 4,
                    # 6422528 = 8192 * 28**2 = vision_tokens * (2*spatial_patch_size)^2
                    "total_pixels": 6422528,
                },
                {"type": "text", "text": f"""You are an autonomous driving safety and perception expert analyzing this video for potential EXTERNAL ANOMALIES that could affect the safe and predictable operation of an autonomous vehicle.

<think>
"CRITICAL ANOMALIES:\n"
- Unexpected obstacles on or near the roadway (debris, animals, fallen objects)
- Pedestrians or cyclists entering or approaching the vehicle’s path
- Vehicles violating traffic rules (red-light running, wrong-way driving, illegal turns)
- Close calls or near-collision situations involving any road user
- Road work zones, blocked lanes, or temporary cones
- Emergency vehicles or flashing lights impacting traffic flow
- Missing, obscured, or malfunctioning traffic signals/signs
- Road surface hazards (potholes, water puddles, ice, uneven terrain)
- Stopped or stalled vehicles obstructing lanes
- Unusual or unpredictable movement of surrounding objects or road users

"CONTEXT MISINTERPRETATION ANOMALIES:\n"
- Situations where the vehicle might misclassify or misinterpret visual cues
    (e.g., a person wearing clothing with a STOP sign print mistaken for a real traffic sign)
- False positives due to reflections, shadows, or advertisements resembling real road objects
- Unclear or deceptive visual context (e.g., temporary paint, digital displays, mirrored surfaces)
- Any environment where perception sensors might interpret context incorrectly and trigger false actions

"OTHER SAFETY CONCERNS:\n"
- Speeding or aggressive driving by surrounding vehicles
- Unsafe lane changes, tailgating, or sudden stops
- Faded or missing lane markings
- Poor visibility (fog, glare, heavy rain, low light)
- Overcrowded intersections or congested roadways
- Objects falling from moving vehicles (cargo, equipment)
- Environmental interference (smoke, dust, reflections)
- Any condition likely to reduce sensor or perception reliability

Analyze the video carefully and identify all external anomalies and context-related perception issues that could cause unsafe or unpredictable autonomous vehicle behavior.
Focus especially on how environmental context could lead to false detection or unsafe reaction.
</think>
                 
<answer>
Is there any external anomaly  in this video? 
</answer>
"""},
                # {"type": "text", "text": "Given the objects in this scene and the contexts in which they are in, is there any behavior, traffic anomaly, or entity that may misguide a naive driving system? Explain your answer in detail."},
            ],
        }
    ]

    # Process inputs
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(conversation)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Run inference
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(SEPARATOR)
    print(output_text[0])
    print(SEPARATOR)


if __name__ == "__main__":
    main()
