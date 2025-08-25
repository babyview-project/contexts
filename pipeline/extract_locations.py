import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
input_dir = '/ccn2/dataset/babyview/outputs_20250312/locations'
output_file = 'merged_locations.csv'

merged_df = pd.read_csv(output_file)
existing_videos = set(merged_df["video_id"])
print(len(existing_videos))
# Collect and process all matching CSV files

for filename in tqdm(os.listdir(input_dir)):
    video_id = Path(filename).stem.removesuffix("_processed")
    if video_id not in existing_videos:
        file_path = os.path.join(input_dir, filename)
        df = pd.read_csv(file_path)
        df['video_id'] = video_id
        merged_df = pd.concat([merged_df, df], ignore_index=True)

# Save merged dataframe
merged_df.to_csv(output_file, index=False)
print(f"Merged CSV saved to: {output_file}")