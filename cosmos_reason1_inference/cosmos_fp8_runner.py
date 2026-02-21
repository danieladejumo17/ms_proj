import os
os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import transformers
import qwen_vl_utils
from sklearn.metrics import precision_score, recall_score, f1_score



_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.metrics import Metrics
from utils.inference_video_writer import annotate_and_write_frames


class CosmosFP8Runner:
    def __init__(self, user_prompt, target_fps: int = 4, target_resolution: tuple[int, int] = (250, 250)):
        self.user_prompt = user_prompt
        self.target_fps = target_fps
        self.target_resolution = target_resolution

        self.load_model("nvidia/Cosmos-Reason1-7B")
        self.warmup_model()

    def run_inference_dataloader(self, dataloader, vid_output, vid_fps, vid_width, vid_height) -> None:
        preds = []
        actuals = []
        inference_times = []

        # ======== Video for the entire inference ============ 
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well
        video_writer = cv2.VideoWriter(vid_output, fourcc, vid_fps, (vid_width, vid_height))
        # ====================================================
        
        # n_samples = (len(ds) - 50)/20 + 1
        for i, sample in enumerate(dataloader): # dataloader yields (video_path, label, new images)
            print(f"[Cosmos_FP8_Runner] Processing video: {sample[0]}")
            infer_start_time = time.time()
            result = self.process_and_run_single_video(sample[0])
            infer_time = time.time() - infer_start_time
            # print(result)
            anomaly = self.parse_result(result)
            actuals.append(sample[1])
            preds.append(anomaly)
            inference_times.append(infer_time)
            
            print(f"[Cosmos_FP8_Runner] \tActual: {sample[1]}, Predicted: {anomaly}, Infer. Time: {infer_time:.2f}s")

            annotate_and_write_frames(
                video_writer, sample[2], anomaly, sample[1], i, vid_width,
            )

        video_writer.release()
        print(f"[Cosmos_FP8_Runner] Processed video for entire Dataloader saved at '{vid_output}'")
        
        self.compute_metrics(preds, actuals)
        self.plot_confusion_matrix(preds, actuals)

        metrics = Metrics()
        # inference_times = [0.0] * len(preds)  # Placeholder for inference times
        metrics.update(preds, actuals, inference_times)
        return metrics

    def load_model(self, model_name: str) -> None:
        print("🔧 Loading and compiling model... This may take a few seconds.")
        start = time.time()

        bnb_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
        self.model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            quantization_config=bnb_config,
            # attn_implementation="flash_attention_2",
            device_map="auto",
            # local_files_only=True
        ).eval()

        self.processor = transformers.AutoProcessor.from_pretrained(model_name) #, local_files_only=True)
        self.model.gradient_checkpointing_disable()
        torch.set_float32_matmul_precision("high")
        self.model = torch.compile(self.model)#, mode="reduce-overhead")

        print(f"✅ Model ready in {time.time() - start:.2f}s\n")
        # return model, processor

    def warmup_model(self) -> None:
        if self.model is None or self.processor is None:
            raise ValueError("Model and processor must be loaded before warmup.")

        print("🔥 Warming up model (compiling kernels)...")
        dummy_conv = [{"role": "user", "content": [{"type": "text", "text": "Is this scene safe?"}]}]
        text = self.processor.apply_chat_template(dummy_conv, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            _ = self.model.generate(**inputs, max_new_tokens=7)
        torch.cuda.synchronize()
        print("✅ Warmup complete.\n")

    def parse_result(self, raw_output: str) -> bool:
        out = raw_output.lower()
        return "anomaly" in out
    
    def _compute_total_pixels(self, video_path):
        """Compute total_pixels budget from the video, matching fp8_inference.py."""
        cap = cv2.VideoCapture(str(video_path))
        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            effective_fps = min(self.target_fps, native_fps) if native_fps > 0 else self.target_fps
            duration = frame_count / native_fps if native_fps > 0 else 0
            num_frames = max(1, int(duration * effective_fps))
        finally:
            cap.release()
        return num_frames * self.target_resolution[0] * self.target_resolution[1]

    def process_and_run_single_video(self, video_path):
        video_path = str(video_path)
        total_pixels = self._compute_total_pixels(video_path)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "total_pixels": total_pixels},
                    {"type": "text", "text": self.user_prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = qwen_vl_utils.process_vision_info(conversation)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=7, do_sample=False)

        new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    
    def compute_metrics(self, predictions, actuals):
        # Accuracy calculation
        correct = sum(p == a for p, a in zip(predictions, actuals))
        accuracy = correct / len(actuals) if actuals else 0
        print("="*30)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        # Computer precision, recall, F1-score
        precision = precision_score(actuals, predictions)
        recall = recall_score(actuals, predictions)
        f1 = f1_score(actuals, predictions)
        print("="*30)
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1-score: {f1 * 100:.2f}%")

        # True Positives, False Positives, True Negatives, False Negatives
        tp = sum((p == 1) and (a == 1) for p, a in zip(predictions, actuals))
        fp = sum((p == 1) and (a == 0) for p, a in zip(predictions, actuals))
        tn = sum((p == 0) and (a == 0) for p, a in zip(predictions, actuals))
        fn = sum((p == 0) and (a == 1) for p, a in zip(predictions, actuals))
        print("="*30)
        print(f"True Positives: {tp}")
        print(f"False Positives: {fp}")
        print(f"True Negatives: {tn}")
        print(f"False Negatives: {fn}")

    def plot_confusion_matrix(self, predictions, actuals):
        # cm = confusion_matrix(actuals, predictions)
        # plt.figure(figsize=(6, 4))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        # plt.xlabel('Predicted Label')
        # plt.ylabel('True Label')
        # plt.title('Confusion Matrix')
        # plt.show()

        
        # tp = sum((p == 1) and (a == 1) for p, a in zip(predictions, actuals))
        # fp = sum((p == 1) and (a == 0) for p, a in zip(predictions, actuals))
        # tn = sum((p == 0) and (a == 0) for p, a in zip(predictions, actuals))
        # fn = sum((p == 0) and (a == 1) for p, a in zip(predictions, actuals))
        # cm = np.array([[tn, fp], [fn, tp]])
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
        # disp.plot(cmap=plt.cm.Blues)
        # plt.title("Confusion Matrix")
        # plt.show()
        pass
