import os
import random
import pandas as pd

TEN_SECOND_DIR = "/ccn2/dataset/babyview/unzip_2025_10s_videos_512p"
# Five videos with the highest location switches based on the frame detections
VIDEOS_OF_INTEREST = ["00320002_2024-08-31_1_4e6440f919_processed", "00370002_2024-04-14_3_14a09138ad_processed", 
                      "00820001_2023-05-11_2_0178226b6b_processed", "00680001_2023-05-13_3_1858c08f1f_processed", 
                      "00370001_2024-01-20_1_28bc1cb38d"]
OUTPUT_PATH = "random_10s_video_paths_0604.csv"

sampled_video_paths = []

# --- Step 2: Randomly sample up to 5 .mp4s from 50 other subdirectories ---
# Get all subdirectories
all_subdirs = [d for d in os.listdir(TEN_SECOND_DIR)
               if os.path.isdir(os.path.join(TEN_SECOND_DIR, d))]

# Exclude the VIDEOS_OF_INTEREST
other_subdirs = [d for d in all_subdirs if d not in VIDEOS_OF_INTEREST]

VIDEOS_OF_INTEREST = VIDEOS_OF_INTEREST + (random.sample(other_subdirs, k=15))

for vid_dir in VIDEOS_OF_INTEREST:
    full_dir_path = os.path.join(TEN_SECOND_DIR, vid_dir)
    if not os.path.isdir(full_dir_path):
        print(f"Directory not found: {full_dir_path}")
        continue

    for file in os.listdir(full_dir_path):
        if file.lower().endswith(".mp4"):
            sampled_video_paths.append(os.path.join(full_dir_path, file))

# Randomly select up to 50 subdirectories
sampled_subdirs = random.sample(other_subdirs, min(50, len(other_subdirs)))

# For each sampled subdir, sample up to 5 .mp4 files
'''
for subdir in sampled_subdirs:
    subdir_path = os.path.join(TEN_SECOND_DIR, subdir)
    mp4_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(".mp4")]
    sampled_files = random.sample(mp4_files, min(5, len(mp4_files)))#

    for file in sampled_files:
        sampled_video_paths.append(os.path.join(subdir_path, file))
'''

# --- Save to CSV ---
df = pd.DataFrame({'video_path': sampled_video_paths})
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(sampled_video_paths)} video paths to {OUTPUT_PATH}")
