import os
import subprocess
from pathlib import Path
from tqdm import tqdm
mp4_folder = "/ccn2/dataset/babyview/unzip_2025/babyview_main_storage"
mp3_folder = "/data/tanawm/bv-transcription/mp3"

def get_duration(file_path):
    """Returns duration in seconds using ffprobe (part of ffmpeg)"""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(file_path)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        return None

diff_lengths = []

for file in tqdm(os.listdir(mp3_folder)):
    if not file.endswith(".mp3"):
        continue

    curr_file_name = Path(file).stem
    curr_dir = curr_file_name.removesuffix("_processed")
    mp4_file_path = Path(f"{mp4_folder}/{curr_dir}/{curr_file_name}.MP4")
    mp3_file_path = Path(mp3_folder) / file

    if not mp4_file_path.exists():
        print(f"Missing MP4: {mp4_file_path}")
        continue

    mp3_length = get_duration(mp3_file_path)
    mp4_length = get_duration(mp4_file_path)

    if mp3_length is None or mp4_length is None:
        continue

    if abs(mp4_length - mp3_length) > 1:  # Tolerance threshold
        diff_lengths.append({
            "file": curr_file_name,
            "mp3_length": mp3_length,
            "mp4_length": mp4_length,
            "difference": mp4_length - mp3_length
        })
        print({
            "file": curr_file_name,
            "mp3_length": mp3_length,
            "mp4_length": mp4_length,
            "difference": mp4_length - mp3_length
        })

# Print mismatches
for diff in diff_lengths:
    print(f"{diff['file']}: MP4 is {diff['difference']:.2f}s {'longer' if diff['difference'] > 0 else 'shorter'}")
