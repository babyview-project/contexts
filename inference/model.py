import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor
import random
import shutil
import ray
import time

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def _set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def get_model_and_processor():
    model_name = "DAMO-NLP-SG/VideoLLaMA3-7B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        # attn_implementation="flash_attention_2",
    )
    model = model.to("cuda")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor


def get_model_response(model, processor, video_path, question):
    start_time = time.time()
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 2, "max_frames": 100}},
                {"type": "text", "text": question},
            ]
        },
    ]

    inputs = processor(conversation=conversation, return_tensors="pt")
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    
    output_ids = model.generate(**inputs, max_new_tokens=512)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(f"Model response time: {time.time() - start_time:.1f} seconds")
    return response

def convert_model_response_to_dict(response, key_list):
    # Check the length of the response
    if len(response) < 10:
        print(f"Response too short, skipping..., response: {response}")
        return None
    
    # Split the response into lines based on the '||' delimiter
    lines = response.split('||')
    lines = [line.strip() for line in lines if line.strip()]
    lines = [line.replace('<', '').replace('>', '') for line in lines]

    # Check that the response contains all required keys and their answers
    response_dict = {}
    for key in key_list:
        found = False
        for line in lines:
            line_split = line.split(":")
            answer = line_split[-1].strip()
            if len(line_split) <= 1 or len(answer) < 2:
                print(f"Answer for {key} too short, skipping..., response: {response}")
                continue
            if line.startswith(key):
                found = True
                response_dict[key] = answer
                break
        if not found:
            print(f"Missing key: {key}, response: {response}")
            return None
    
    # Join the lines back together
    return response_dict