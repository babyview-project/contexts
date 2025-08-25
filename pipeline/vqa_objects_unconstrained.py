import pandas as pd
from models.video.videochat_flash import VideoFlash
from ray.experimental.tqdm_ray import tqdm
#from models.image.internvl import InternVL
import ray
from datetime import datetime
from pathlib import Path
import argparse
import os
import random
import re

df = []
num_processes=6
overall_video_dir = '/ccn2/dataset/babyview/unzip_2025_10s_videos_256p/'
output_dir = '/ccn2/dataset/babyview/outputs_20250312/yoloe/unconstrained/'
recordings_path = '/home/tsepuri/activitycap/data/recordings_processed.csv'
@ray.remote(num_gpus=1)
def run_unconstrained_objects(curr_videos, output_dir):
    video_generator = VideoFlash()
    recordings = pd.read_csv(recordings_path)
    for dir_num, video_dir in tqdm(enumerate(curr_videos), total=len(curr_videos)):
        video_basename = os.path.basename(video_dir)
        superseded_id = re.sub(r"_processed", "", video_basename).strip()
        out_df_path = Path(output_dir) / f"{video_basename}.csv"
        home_video_ids = set(recordings["superseded_gcp_name_feb25"])
        print(len(home_video_ids))
        if os.path.exists(out_df_path) or (not superseded_id in home_video_ids and "_H" not in superseded_id):
            print(superseded_id)
            continue
        video_files = [f for f in os.listdir(video_dir)]
        if len(video_files) == 0:
            continue
        video_files = sorted(video_files)
        video_files = video_files[::3]
        vid_results = []
        for vid_num, video_path in enumerate(video_files):
            video_path = os.path.join(video_dir, video_path)
            objects, _ = video_generator.caption_video(video_path,f"This is a video from the point-of-view of a camera mounted on a child's head. Strictly return a list detailing each object, animal and person present in this video, comma separated like so: 'ball,tennis racquet,person,sofa'")
            objects = objects.strip()
            vid_results.append({
                "video_id": video_path,
                "objects": objects
            })
        out_df = pd.DataFrame(vid_results).sort_values(by='video_id').reset_index(drop=True) # sort by video_id
        out_df.to_csv(out_df_path, index=False)

def main():
    parser = argparse.ArgumentParser(description='Get object detections across clips')
    parser.add_argument('--input_dir', default=overall_video_dir, help='Path to input video directory')
    parser.add_argument('--output_dir', '-o', 
                      help='Path to output CSV file',
                      default=output_dir)
    
    args = parser.parse_args()
    ray.init(num_cpus = num_processes)
    all_video_dirs = [os.path.join(overall_video_dir, d) for d in os.listdir(overall_video_dir) if os.path.isdir(os.path.join(overall_video_dir, d))]
    random.shuffle(all_video_dirs)
    print(f"Total video directories: {len(all_video_dirs)}")
    chunk_size = len(all_video_dirs) // num_processes + (1 if len(all_video_dirs) % num_processes else 0)
    video_chunks = [all_video_dirs[i:i+chunk_size] for i in range(0, len(all_video_dirs), chunk_size)]
    futures = [run_unconstrained_objects.remote(chunk, args.output_dir) for i, chunk in enumerate(video_chunks)]
    ray.get(futures)  
    
if __name__ == "__main__":
    main()