import os
import numpy as np
import pandas as pd
import torch
import random
import shutil
import ray
import re
from models.video.videollama_constrained import VideoLLaMA
from models.video.vqa import _set_seed
from constants import get_activities, get_locations, get_bv_main_ids
import argparse
from tqdm import tqdm

"""
cd /home/tsepuri/contexts/inference
python constrained_questions.py
"""

# transformers==4.51.1
_set_seed(42)

num_processes = 8

overall_video_dir = '/home/tsepuri/contexts/test_chunks/' #'/ccn2/dataset/babyview/unzip_2025_10s_videos_256p/'

prompt_key_values = {
    "Location": get_locations(),
    "Activity": get_activities(),
    "Video description": None
}

def create_question():
    question = "This video is recorded from the point-of-view of a child, with a camera mounted on the child's head. Respond strictly only in this format with both keys and values: "
    for key, values in prompt_key_values.items():
        if values is None:
            prompt_value = "..."
        else:
            prompt_value =  f"<{'/'.join(values)}>"
        question += f"{key}: {prompt_value} || "
    return question

@ray.remote(num_gpus=1)
def get_model_responses_for_video_sublist(video_dir_sublist, chunk_id, out_vis_dir, side_by_side_vis_dir, overwrite):
    question = create_question()
    prompter = VideoLLaMA(use_constrained_decoding=True)
    
    # Use chunk_id for better progress tracking
    pbar_dirs = tqdm(enumerate(video_dir_sublist), 
                     total=len(video_dir_sublist),
                     desc=f"Chunk {chunk_id} - Processing directories",
                     leave=False)
    
    for dir_num, video_dir in pbar_dirs:
        out_df_path = os.path.join(out_vis_dir, os.path.basename(video_dir) + '.csv')
        if os.path.exists(out_df_path) and not overwrite:
            print(f"Output file {out_df_path} already exists. Skipping...")
            continue

        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        if len(video_files) == 0:
            continue
        random.shuffle(video_files)

        out_df = pd.DataFrame(columns=['video_id'] + list(prompt_key_values.keys()))
        
        # Update description for current directory
        pbar_dirs.set_description(f"Chunk {chunk_id} - Dir {dir_num+1}/{len(video_dir_sublist)} ({len(video_files)} videos)")
        
        # Progress bar for videos within current directory
        pbar_videos = tqdm(enumerate(video_files), 
                          total=len(video_files),
                          desc=f"Processing videos in {os.path.basename(video_dir)}",
                          leave=False)
        
        for vid_num, video_path in pbar_videos:
            video_path = os.path.join(video_dir, video_path)
            # Update video progress description
            pbar_videos.set_description(f"Video {vid_num+1}/{len(video_files)}: {os.path.basename(video_path)[:20]}...")
            try:
                response = None
                response_dict = None
                try_count = 0
                max_try_count = 5

                while response is None or response_dict is None:
                    try_count += 1
                    if try_count > 1:
                        # changing question slightly to see if that helps the model process the prompt differently 
                        question = question
                    if try_count > max_try_count:
                        print(f"Failed to get response after {max_try_count} tries, skipping video: {video_path}")
                        break
                    response = prompter.get_response_with_constraints(video_path, question, prompt_key_values)
                    response_dict = prompter.convert_model_response_to_dict(response, list(prompt_key_values.keys()), prompt_key_values)
                    #response = prompter.get_response(video_path, question)
                    #response_dict = prompter.convert_model_response_to_dict(response, list(prompt_key_values.keys()), prompt_key_values)
                
                # adding an empty dict if no response generated
                if response_dict is None:
                    print(f"Empty response for {video_path} after {try_count} tries, adding empty dict")
                    response_dict = {key: "" for key in prompt_key_values.keys()}

                # Append to output dataframe
                response_dict['video_id'] = os.path.basename(video_path)
                response_dict['superseded_gcp_name_feb25'] = re.sub(r"_processed*", "", video_path)
                response_dict['video_path'] = video_path
                out_df = pd.concat([out_df, pd.DataFrame([response_dict])], ignore_index=True)
                out_df['video_id'] = out_df['video_id'].astype(str)

                if dir_num < 20 and vid_num == 0:
                    # Save outputs: Video and Model response
                    video_basename = os.path.basename(video_path).split('.')[0]
                    out_video_path = os.path.join(side_by_side_vis_dir, video_basename + '.mp4')
                    os.system(f'cp {video_path} {out_video_path}')

                    out_model_response_path = os.path.join(side_by_side_vis_dir, video_basename + '.txt')
                    with open(out_model_response_path, 'w') as f:
                        for key in prompt_key_values.keys():
                            f.write(f"{key}: {response_dict[key]}\n")
                        f.write("===== \nQuery: " + question)
                        
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")
        
        pbar_videos.close()
        
        out_df = out_df.sort_values(by='video_id').reset_index(drop=True)

        out_df.to_csv(out_df_path, index=False)
    
    pbar_dirs.close()
    return f"Chunk {chunk_id} completed"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default=None, help='Path to specific video file')
    parser.add_argument('--video_dir', type=str, default=overall_video_dir, help='Path to directory containing videos')
    parser.add_argument('--output_path', type=str, default="../predictions/", help='Path to save outputs')
    parser.add_argument('--num_processes', type=int, default=8, help='Number of parallel processes')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output files')
    args = parser.parse_args()
    num_processes = args.num_processes
    bv_main_ids = get_bv_main_ids()
    print(len(bv_main_ids))
    if args.video_path:
        # Process single video
        video_dirs = [os.path.dirname(args.video_path)]
    else:
        # Process all videos in directory and subdirectories
        video_dirs = [os.path.join(args.video_dir, d) for d in os.listdir(args.video_dir) 
                     if os.path.isdir(os.path.join(args.video_dir, d)) and re.sub(r"(_processed*)", "", d, flags=re.IGNORECASE) in bv_main_ids]
        # Add main directory if it contains videos
        if any(f.endswith('.mp4') for f in os.listdir(args.video_dir)):
            video_dirs.append(args.video_dir)
        random.shuffle(video_dirs)

    out_vis_dir = os.path.join(args.output_path, 'videollama3_constrained')
    side_by_side_vis_dir = os.path.join(out_vis_dir, 'side_by_side')
    os.makedirs(out_vis_dir, exist_ok=True)
    os.makedirs(side_by_side_vis_dir, exist_ok=True)
    
    print(f"Total video directories: {len(video_dirs)}")
       
    # Split video_dirs into chunks for parallel processing
    ray.init()
    chunk_size = len(video_dirs) // num_processes + (1 if len(video_dirs) % num_processes else 0)
    video_chunks = [video_dirs[i:i+chunk_size] for i in range(0, len(video_dirs), chunk_size)]
    
    print(f"Processing {len(video_chunks)} chunks with {num_processes} processes...")
    
    # Run parallel tasks with chunk IDs
    futures = [get_model_responses_for_video_sublist.remote(chunk, i, out_vis_dir, side_by_side_vis_dir, args.overwrite) 
               for i, chunk in enumerate(video_chunks)]
    # Wait for completion with overall progress
    print("Waiting for all chunks to complete...")
    results = ray.get(futures)
    print("All processing complete!")
