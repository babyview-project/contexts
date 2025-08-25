from pathlib import Path
import pandas as pd
from tqdm import tqdm
from video_utils import split_video, batcher
import argparse

parser = argparse.ArgumentParser(description="Chunk videos using ffmpeg.")
parser.add_argument("--rank_id", type=int, default=1, help="Batch number of total batches being run, one-indexed.")
parser.add_argument("--num_parallel", type=int, default=1, help="Total batches being run in parallel.")
parser.add_argument("--input", type=str, default="all_video_paths.csv", help="Input csv for chunking")
parser.add_argument("--output_dir", type=str, default="/ccn2/dataset/babyview/outputs_20250312/activities/chunks")
args = parser.parse_args()
df = pd.read_csv(args.input)
sampled_video_paths = batcher(df["video_path"], args.rank_id+1, args.num_parallel)

for vid_path in tqdm(sampled_video_paths, desc="Chunking videos"):
    curr_video_id = Path(vid_path).stem
    split_video(vid_path, 60, Path(args.output_dir) / curr_video_id)
