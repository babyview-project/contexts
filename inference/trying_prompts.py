import argparse
import os
import numpy as np
import pandas as pd
import torch
import random
import shutil
import ray

from model import _set_seed, get_model_and_processor, get_model_response, convert_model_response_to_dict

"""
In-progress (!) code for using a VQA model to get unconstrained labels for categories from videos.

cd /ccn2/u/khaiaw/Code/babyview-pose/contexts/inference
conda activate babyview-pose

CUDA_VISIBLE_DEVICES=6 \
python trying_prompts.py \
    --input_video_list_file bv_activities_groundtruth_clips.txt \
    --base_prompt_num V3 \
    --annotation_type child_activity \
    --annotation_num V1 \
    --out_vis_dir ./vis_model_predictions/2025_10_24/ \
    --num_processes 1 \

"""


annotation_type_to_key_list = {
    'child_activity': 'Child activity',
    'adult_activity': 'Adult activity',
    'posture': 'Posture',
}

base_prompts = {
    "V3": "This is a video from a camera mounted on the head of a young child. ",
    "V4": "Imagine that you are a young child, and this video reflects what you are seeing and doing. ",
    "V5": "Strictly consider the camera as being worn by a young child at home. ",
}

annotation_dict = {
    'child_activity': {
        'V1': 'What activity is the child doing?',
        'V2': 'What activity is the wearer of the camera doing?',
    },
    'adult_activity': {
        'V1': 'What activity is the adult(s) doing? If there are no adults, respond with none',
        'V2': 'What is the general activity that the other people in the scene are doing? If there are no other people, respond with none',
    },
    'posture': {
        'V1': 'What is the body posture of the child?',
        'V2': 'What is the body posture of the wearer of the camera?',
    }
}

def create_question(base_prompt_num, annotation_type, annotation_num):
    # question = "This a video from the point-of-view of a camera mounted on a child's head. Respond strictly only in this format with both keys and values: "
    # question = "This is a video from a camera mounted on the head of a young child. "
    question = ""
    question += f'{base_prompts[base_prompt_num]} '
    question += f'{annotation_dict[annotation_type][annotation_num]} '

    question += "Respond strictly only in this format with this key and value: "
    key_list = [annotation_type_to_key_list[annotation_type]]
    for key in key_list:
        question += f"{key}: ... || "
    return question

@ray.remote(num_gpus=1)
def get_model_responses_for_video_sublist(video_path_list, base_prompt_num, annotation_type, annotation_num, vis_dir):
    model, processor = get_model_and_processor()
    
    question = create_question(base_prompt_num, annotation_type, annotation_num)
    key_list = [annotation_type_to_key_list[annotation_type]]

    vis_dir = os.path.join(vis_dir, f'base_{base_prompt_num}_{annotation_type}_annotation_{annotation_num}')
    for video_path in video_path_list:
        _ = call_model_response(model, processor, video_path, question, key_list, vis_dir)
            
def call_model_response(model, processor, video_path, question, key_list, vis_dir):
    try:
        response = None
        try_count = 0
        max_try_count = 5
        while response is None:
            try_count += 1
            if try_count > max_try_count:
                print(f"Failed to get response after {max_try_count} tries, skipping video_path: {video_path}")
                break
            response = get_model_response(model, processor, video_path, question)
            response_dict = convert_model_response_to_dict(response, key_list)
        
        if response_dict is None:
            return
        
            # Save outputs: Video and Model response
        video_basename = os.path.basename(video_path).split('.')[0]
        out_video_path = os.path.join(vis_dir, video_basename + '.mp4')
        os.makedirs(vis_dir, exist_ok=True)
        os.system(f'cp {video_path} {out_video_path}')

        out_model_response_path = os.path.join(vis_dir, video_basename + '.txt')
        with open(out_model_response_path, 'w') as f:
            for key in key_list:
                f.write(f"{key}: {response_dict[key]}\n")
            f.write("===== \nQuery: " + question)
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_processes', type=int, default=1, help='Number of parallel processes to use')
    parser.add_argument('--base_prompt_num', type=str, default='V3', help='Base prompt version to use')
    parser.add_argument('--annotation_type', type=str, default='child_activity', help='Type of annotation to query')
    parser.add_argument('--annotation_num', type=str, default='V1', help='Annotation prompt version to use')
    parser.add_argument('--out_vis_dir', type=str, default='./vis_model_predictions/2025_10_24/', help='Output directory for visualizations and model responses')
    parser.add_argument('--input_video_list_file', type=str, help='Optional file containing list of video directories to process')
    args = parser.parse_args()
    
    # read the input_video_list_file
    if args.input_video_list_file is not None:
        with open(args.input_video_list_file, 'r') as f:
            video_paths = [line.strip() for line in f.readlines()]

    # Run parallel tasks
    ray.init()
    chunk_size = len(video_paths) // args.num_processes + (1 if len(video_paths) % args.num_processes else 0)
    video_chunks = [video_paths[i:i+chunk_size] for i in range(0, len(video_paths), chunk_size)]

    futures = [get_model_responses_for_video_sublist.remote(chunk, args.base_prompt_num, args.annotation_type, args.annotation_num, args.out_vis_dir) for chunk in video_chunks]
    ray.get(futures)
    
    