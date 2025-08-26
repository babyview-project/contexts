import logging
import time
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor
from models.video.vqa import VQA
import torch
class VideoLLaMA(VQA):
    def __init__(self, model_name: str = "DAMO-NLP-SG/VideoLLaMA3-7B", device:str="cuda"):
        super().__init__(model_name, device)

    def _setup_model(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        self.max_frames = 100
        self.fps = 2
    
    def get_response(self, video_path, question: str) -> str:
        start_time = time.time()
        video_path = video_path
        fps = self.fps if self.fps is not None else 2
        max_frames = self.max_frames if self.max_frames is not None else 100

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": fps, "max_frames": max_frames}},
                    {"type": "text", "text": question},
                ]
            },
        ]

        inputs = self.processor(conversation=conversation, return_tensors="pt")
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        
        output_ids = self.model.generate(**inputs, max_new_tokens=512)
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        logging.info(f"Model response time: {time.time() - start_time:.1f} seconds")
        return response
