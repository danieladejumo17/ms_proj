#!/usr/bin/env python3
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

"""Single-image inference helper for Cosmos-Reason1."""

from __future__ import annotations

import argparse
from pathlib import Path

import qwen_vl_utils
import transformers

SEPARATOR = "-" * 20

SYSTEM_PROMPT = (
    "You are an autonomous driving perception safety analyst. "
    "Perform disciplined, high-recall reasoning about semantic and contextual anomalies "
    "that could mislead a naive autonomous vehicle."
)

USER_PROMPT = """<think>
Inspect the road scene carefully. Focus on semantic anomalies: objects, signage, lane markings, or context cues that look normal to humans but could cause incorrect perception or planning for an autonomous vehicle.
- Note mismatches between objects and their expected location or behavior.
- Highlight deceptive cues (reflections, posters, unusual attire, digital displays) that could trigger false detections.
- Consider whether the scene hides true hazards (pedestrians, cyclists, vehicles, obstacles) that may be misclassified.
- Evaluate lighting, weather, occlusions, or sensor limitations that amplify confusion.
</think>

<answer>
Provide a short safety report using this template:
Final Assessment:
Anomaly: {Yes|No}
Evidence: <key cues and where they appear>
Impact: <how a naive autonomous vehicle could respond poorly>
Mitigation: <recommended safe fallback or verification>

Replace {Yes|No} with either Yes or No. Do not include braces or placeholders. Always include the line starting with "Anomaly: ".
</answer>
"""


def build_conversation(
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
) -> list[dict]:
    """Construct the multimodal conversation for the model."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the image file to analyze",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Reason1-7B",
        help="Model name or local path",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        help="Optional path to a file containing a custom system prompt",
    )
    parser.add_argument(
        "--user-prompt-file",
        type=Path,
        help="Optional path to a file containing a custom user prompt",
    )
    args = parser.parse_args()

    image_path = args.image.expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    system_prompt = SYSTEM_PROMPT
    if args.system_prompt_file is not None:
        system_prompt_path = args.system_prompt_file.expanduser()
        if not system_prompt_path.exists():
            raise FileNotFoundError(f"System prompt file not found: {system_prompt_path}")
        system_prompt = system_prompt_path.read_text(encoding="utf-8")

    user_prompt = USER_PROMPT
    if args.user_prompt_file is not None:
        user_prompt_path = args.user_prompt_file.expanduser()
        if not user_prompt_path.exists():
            raise FileNotFoundError(f"User prompt file not found: {user_prompt_path}")
        user_prompt = user_prompt_path.read_text(encoding="utf-8")

    conversation = build_conversation(image_path, system_prompt, user_prompt)

    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    processor: transformers.Qwen2_5_VLProcessor = (
        transformers.AutoProcessor.from_pretrained(args.model)
    )

    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
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

    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
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
