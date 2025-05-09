from pathlib import Path
import pandas as pd
from tqdm import tqdm
from video_utils import split_video_fast
import argparse

df = pd.read_csv("random_video_paths.csv")
parser = argparse.ArgumentParser(description="Chunk videos using ffmpeg.")
parser.add_argument("--batch", type=int, default=1, help="Batch number of total batches being run, one-indexed.")
parser.add_argument("--total_batches", type=int, default=1, help="Total batches being run in parallel.")
args = parser.parse_args()
total_paths = len(df)
batch_size = total_paths // args.total_batches
remainder = total_paths % args.total_batches

if args.batch <= remainder:
    # First 'remainder' batches get one extra item
    start = (batch_size + 1) * (args.batch - 1)
    end = start + (batch_size + 1)
else:
    # Remaining batches get standard batch_size
    start = (batch_size + 1) * remainder + batch_size * (args.batch - remainder - 1)
    end = start + batch_size

sampled_video_paths = df["video_path"].iloc[start:end]
for vid_path in tqdm(sampled_video_paths, desc="Chunking videos"):
    curr_video_id = Path(vid_path).stem
    split_video_fast(vid_path, 60, f"/ccn2/dataset/babyview/outputs_20250312/activities/chunks/{curr_video_id}")
