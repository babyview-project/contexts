# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
from models.video_captioner import VideoCaptioner
import sys
import warnings
warnings.filterwarnings("ignore")
class LLAVAOneVision(VideoCaptioner):
    def __init__(self, device="cuda"):
        pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov-chat"
        self.model_name = "llava_qwen"
        device_map = "auto"
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(pretrained, None, self.model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args
        self.model.eval()
        torch.cuda.empty_cache()
        super().__init__(device)

    def create_question(self, question, template="qwen_1_5"):
        question = DEFAULT_IMAGE_TOKEN + f"\n{question}"
        conv_template = template
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        return input_ids


    def caption_images(self, image_paths=["https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"], text="What is shown in this image?"):
        images = self.open_images(image_paths)
        image_tensor = process_images(images, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
        image_sizes = [img.size for img in images]
        cont = self.model.generate(
            self.create_question(text),
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=300,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        return text_outputs
