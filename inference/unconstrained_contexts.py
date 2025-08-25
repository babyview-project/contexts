import os
import numpy as np
import pandas as pd
import torch
import random
import shutil
import ray
from inference.models.video.videollama import VideoLLaMA

"""
cd /ccn2/u/khaiaw/Code/babyview-pose/object-detection/video-qa
python unconstrained_questions.py
In-progress (!) code for using a VQA model to get unconstrained labels for categories from videos.
"""

prompter = VideoLLaMA()
prompter._set_seed(42)

num_processes = 8

overall_video_dir = '/ccn2/dataset/babyview/unzip_2025_10s_videos_256p/'
out_vis_dir = './vis_model_predictions/'
out_vis_dir = os.path.join(out_vis_dir, 'unconstrained_activities')
side_by_side_vis_dir = os.path.join(out_vis_dir, 'side_by_side')
if os.path.exists(out_vis_dir):
    shutil.rmtree(out_vis_dir)
os.makedirs(out_vis_dir, exist_ok=True)
os.makedirs(side_by_side_vis_dir, exist_ok=True)

key_list = [
    'Location',
    'Activity',
]

def create_question():
    question = "This a video from the point-of-view of a camera mounted on a child's head. Respond strictly only in this format with both keys and values: "
    for key in key_list:
        question += f"{key}: ... || "
    return question

@ray.remote(num_gpus=1)
def get_model_responses_for_video_sublist(video_dir_sublist):
    question = create_question()
    
    
    for dir_num, video_dir in enumerate(video_dir_sublist):
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        if len(video_files) == 0:
            continue
        
        video_files = sorted(video_files)
        # indices = np.linspace(0, len(video_files) - 1, min(20, len(video_files)), dtype=int) # select 20 evenly spaced indices
        # video_files = [video_files[i] for i in indices]
        video_files = video_files[::3] # sample every 3rd video
        random.shuffle(video_files) # shuffle the selected videos
        
        out_df = pd.DataFrame(columns = ['video_id'] + key_list)
        for vid_num, video_path in enumerate(video_files):
            video_path = os.path.join(video_dir, video_path)
            try:
                response = None
                try_count = 0
                max_try_count = 3
                while response is None:
                    try_count += 1
                    if try_count > max_try_count:
                        print(f"Failed to get response after {max_try_count} tries, skipping video_dir: {video_dir}")
                        break
                    response = prompter.get_model_response(prompter.model, prompter.processor, video_path, question)
                    response_dict = prompter.convert_model_response_to_dict(response, key_list)
                
                if response_dict is None:
                    continue
                
                # Append to output dataframe
                response_dict['video_id'] = os.path.basename(video_path)
                out_df = pd.concat([out_df, pd.DataFrame([response_dict])], ignore_index=True)
                out_df['video_id'] = out_df['video_id'].astype(str)
                
                if dir_num < 20 and vid_num == 0:
                    # Save outputs: Video and Model response
                    video_basename = os.path.basename(video_path).split('.')[0]
                    out_video_path = os.path.join(side_by_side_vis_dir, video_basename + '.mp4')
                    os.system(f'cp {video_path} {out_video_path}')

                    out_model_response_path = os.path.join(side_by_side_vis_dir, video_basename + '.txt')
                    with open(out_model_response_path, 'w') as f:
                        for key in key_list:
                            f.write(f"{key}: {response_dict[key]}\n")
                        f.write("===== \nQuery: " + question)
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")
        
        out_df = out_df.sort_values(by='video_id').reset_index(drop=True) # sort by video_id
        out_df_path = os.path.join(out_vis_dir, os.path.basename(video_dir) + '.csv')
        out_df.to_csv(out_df_path, index=False)

if __name__ == "__main__":
    # For each directory, randomly select a video file
    video_dirs = [os.path.join(overall_video_dir, d) for d in os.listdir(overall_video_dir) if os.path.isdir(os.path.join(overall_video_dir, d))]
    random.shuffle(video_dirs)
    print(f"Total video directories: {len(video_dirs)}")
       
    # Split video_dirs into chunks for parallel processing
    ray.init()
    chunk_size = len(video_dirs) // num_processes + (1 if len(video_dirs) % num_processes else 0)
    video_chunks = [video_dirs[i:i+chunk_size] for i in range(0, len(video_dirs), chunk_size)]
    
    # Run parallel tasks
    futures = [get_model_responses_for_video_sublist.remote(chunk) for chunk in video_chunks]
    ray.get(futures)
    
    