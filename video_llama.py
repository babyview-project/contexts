import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor
import random
import shutil
import ray

"""
In-progress (!) code for using a VQA model to analyze videos from the BabyView dataset.
"""

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

num_processes = 2

overall_video_dir = '/ccn2/dataset/babyview/unzip_2025_10s_videos_256p/'
out_vis_dir = './vis_model_predictions/'
out_vis_dir = os.path.join(out_vis_dir, 'object_interaction')
if os.path.exists(out_vis_dir):
    shutil.rmtree(out_vis_dir)
os.makedirs(out_vis_dir, exist_ok=True)

# key_list = ['Child Position']
key_list = []

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

def get_model_response(model, processor, video_path):
    question = "This a video from the point-of-view of a head-mounted camera on a child. Respond strictly only in this format with both keys and values: "
    question += "Environment: <Indoors/Outdoors> || "
    question += "Location: <Kitchen/Dining room/Living room/Bedroom/School/Park/Backyard> || "
    question += "Activity: <Playing/Eating/Reading/Relaxing/Watching/Walking> || "
    # question += "Sub-Location: <> || "
    # question += "Person(s): [...] || "
    
    # question += "Child Position: <Sitting or standing on ground/Sitting on couch or table or other furniture/Held by adult> || "

    # question += "Provide a short explanation about what the camera-wearing child is sitting or standing on and their position relative to the ground or furniture || "
    # question += "Child is on the: <Ground/Furniture> || "
    # question += "Sitting or standing on the ground: <Yes/No> || "
    # question += "Sitting on some furniture: <Yes/No> || "
    # question += "Held by an adult: <Yes/No> || "
    # question += "Is there someone gazing at the camera: <Yes/No> || "
    # question += "Posture: <Walking/Crawling/Sitting/Standing>"
    question += "Object child is interacting with: <___/None>"
    


    # question += "Adult Pointing Hand Visible: <True/False>"
    # question += "Baby Posture: <Lying/Sitting/Crawling/Standing/Walking/Held> || "
    # question += "Camera Motion: <Still/Moving> || "

    # question += "Child Position: <Ground/Above ground> || "
    # question += "Child Posture: <Sitting/Standing/Held by adult> || "
    
    # question += "Adult interacting with child: <True/False> || "
    # question += "Nearby Focal Objects: () || "
    # question += "Object Size: <Small/Medium/Large> || "
    # question += "Adult Gaze Direction: <Toward Baby/Toward Object/Elsewhere/Unknown> || "
    # question += "Pointing Hand Visible: <True/False>"

    # Video conversation
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
    return response, question

def sanitize_model_response(response):
    # Check the length of the response
    if len(response) < 10:
        print(f"Response too short, skipping..., response: {response}")
        return None
    
    # if 'indoor' in response.lower() and 'outdoor' not in response.lower():
    #     print(f"Response is indoor', skipping..., response: {response}")
    #     return None
    
    # if 'at the camera: Yes' not in response:
    #     return None
    
    # Split the response into lines based on the '||' delimiter
    lines = response.split('||')
    lines = [line.strip() for line in lines if line.strip()]
    lines = [line.replace('<', '').replace('>', '') for line in lines]
    
    # Check that the response contains all required keys and their answers
    for key in key_list:
        found = False
        for line in lines:
            line_split = line.split(":")
            answer = line_split[-1].strip()
            if len(line_split) <= 1 or len(answer) < 2:
                print(f"Answer for {key} too short, skipping..., response: {response}")
                return None
            if line.startswith(key):
                found = True
                break
        if not found:
            print(f"Missing key: {key}, response: {response}")
            return None
    
    # Join the lines back together
    lines.insert(3, "==================")
    ret = '\n'.join(lines)
    return ret    

@ray.remote(num_gpus=1)
def get_model_responses_for_video_sublist(video_dir_sublist):
    model, processor = get_model_and_processor()
    for video_dir in video_dir_sublist:
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        if len(video_files) == 0:
            continue
        
        try:
            response = None
            try_count = 0
            max_try_count = 1
            while response is None:
                try_count += 1
                if try_count > max_try_count:
                    print(f"Failed to get response after {max_try_count} tries, skipping video_dir: {video_dir}")
                    break
                video_path = os.path.join(video_dir, random.choice(video_files))
                response, question = get_model_response(model, processor, video_path)
                response = sanitize_model_response(response)
            
            if response is None:
                continue
            
            # Save outputs: Video and Model response
            video_basename = os.path.basename(video_path).split('.')[0]
            out_video_path = os.path.join(out_vis_dir, video_basename + '.mp4')
            os.system(f'cp {video_path} {out_video_path}')

            out_model_response_path = os.path.join(out_vis_dir, video_basename + '.txt')
            with open(out_model_response_path, 'w') as f:
                f.write(response)
                f.write("===== \nQuery:" + question)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")

if __name__ == "__main__":
    # For each directory, randomly select a video file
    video_dirs = [os.path.join(overall_video_dir, d) for d in os.listdir(overall_video_dir) if os.path.isdir(os.path.join(overall_video_dir, d))]
    random.shuffle(video_dirs)
       
    # Split video_dirs into chunks for parallel processing
    ray.init(num_cpus=num_processes)
    chunk_size = len(video_dirs) // num_processes + (1 if len(video_dirs) % num_processes else 0)
    video_chunks = [video_dirs[i:i+chunk_size] for i in range(0, len(video_dirs), chunk_size)]
    
    # Run parallel tasks
    futures = [get_model_responses_for_video_sublist.remote(chunk) for chunk in video_chunks]
    ray.get(futures)
    
    