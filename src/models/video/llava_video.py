from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from video_utils import load_video
import numpy as np
from models.video_captioner import VideoCaptioner
warnings.filterwarnings("ignore")

class LLAVAVideo(VideoCaptioner):
    def __init__(self, new_tokens=200):
        self.pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
        self.model_name = "llava_qwen"
        self.device = "cuda"
        self.device_map = "auto"
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(self.pretrained, None, self.model_name, torch_dtype="bfloat16", device_map=self.device_map)  # Add any other thing you want to pass in llava_model_args
        self.model.eval()
        self.max_frames_num = 64
        self.fps = 1
        self.conv_template = "qwen_1_5"
        self.max_new_tokens = 200

    def caption_video(self, video_path="/ccn2/dataset/babyview/unzip_2025/babyview_main_storage/00220001_2024-05-31_1_acd11db79d/00220001_2024-05-31_1_acd11db79d_processed.MP4", question="Describe this video in detail."):
        video,frame_time,video_time = load_video(video_path, max_frames_num=self.max_frames_num, fps=self.fps, force_sample=True)
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
        video = [video]
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\nPlease describe this video in detail."
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        cont = self.model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=self.max_new_tokens,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return text_outputs
