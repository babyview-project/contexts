from transformers import AutoModel, AutoTokenizer
import torch
from dotenv import load_dotenv
import os
from models.video_captioner import VideoCaptioner

load_dotenv()

token = os.environ['HF_TOKEN']
class VideoFlash(VideoCaptioner):
    def __init__(self):
        # model setting
        model_path = 'OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
        self.image_processor = self.model.get_vision_tower().image_processor
        mm_llm_compress = False # use the global compress or not
        if mm_llm_compress:
            self.model.config.mm_llm_compress = True
            self.model.config.llm_compress_type = "uniform0_attention"
            self.model.config.llm_compress_layer_list = [4, 18]
            self.model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
        else:
            self.model.config.mm_llm_compress = False
        self.generation_config = dict(
                do_sample=False,
                temperature=0.0,
                max_new_tokens=300,
                top_p=0.1,
                num_beams=1
        )
        # evaluation setting
        self.max_num_frames = 1000

    def caption_video(self, video_path="/ccn2/dataset/babyview/unzip_2025/babyview_main_storage/00220001_2024-05-31_1_acd11db79d/00220001_2024-05-31_1_acd11db79d_processed.MP4", question="Describe this video in detail."):
        output, chat_history = self.model.chat(video_path=video_path, tokenizer=self.tokenizer, user_prompt=question, return_history=True, max_num_frames=self.max_num_frames, generation_config=self.generation_config)
        return output, chat_history
