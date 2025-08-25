from pathlib import Path
import pandas as pd
import random
import re
from tqdm import tqdm

# Paths
BASE_DIR = Path("/ccn2/dataset/babyview/outputs_20250312")
RANDOM_VIDEO_PATHS = "random_10s_video_paths_0604.csv"
ALL_VIDEO_PATHS = "all_video_paths.csv"
CHUNKS_DIR = BASE_DIR / "activities/chunks"
ALREADY_CHUNKED = True
TRANSCRIPTS_DIR = BASE_DIR / "transcripts/diarised"
OUTPUT_CSV = "selected_chunk_transcripts_0604.csv"
CHUNK_SIZE = 10
VIDEOS_DIR = "/ccn2/dataset/babyview/unzip_2025/babyview_main_storage"

def main():
    # Load video paths
    video_paths = pd.read_csv(RANDOM_VIDEO_PATHS)['video_path'].tolist()
    
    selected_chunks = []
    selected_chunk_ids = set()
    
    # Keep selecting until we have 100 unique chunks
    for video_path in tqdm(video_paths):
    #while len(selected_chunks) < 1000 and not ALREADY_CHUNKED:
        # Select random video
    #    if not ALREADY_CHUNKED:
        #video_path = random.choice(video_paths)
        videostem = Path(video_path).stem
        videostem = re.sub(r'(_processed)?_\d+$', '', videostem)
        subject_id = videostem.split('_')[0]
        if ALREADY_CHUNKED:
            chunk_file = video_path
            video_path = Path(VIDEOS_DIR) / videostem / video_path
            chunk_id = int(re.search(r'(\d+)\.mp4$', Path(chunk_file).name).group(1))
        else:
            # Find chunks for this video
            chunk_dir = CHUNKS_DIR / Path(video_path).stem
            if not chunk_dir.exists():
                continue
                
            chunk_files = list(chunk_dir.glob("*.mp4"))
            if not chunk_files:
                continue
                
            # Select random chunk
            chunk_file = random.choice(chunk_files)
            chunk_id = int(re.search(r'(\d+)\.mp4$', chunk_file.name).group(1))
            unique_chunk_id = chunk_file.parent.name + '/' + chunk_file.name
            # Skip if we've already selected this chunk
            if unique_chunk_id in selected_chunk_ids:
                continue
            
        # Get transcript
        transcript_path = TRANSCRIPTS_DIR / subject_id / f"{videostem}.csv"
        if not transcript_path.exists():
            print(transcript_path)
            continue
            
        # Process transcript
        try:
            transcript_df = pd.read_csv(transcript_path)
            
            # Time range for this chunk
            start_time = chunk_id * CHUNK_SIZE
            end_time = start_time + CHUNK_SIZE
            def time_to_seconds(time_str):
                if isinstance(time_str, str):
                    parts = time_str.split(':')
                    if len(parts) == 3:
                        h, m, s = map(float, parts)
                        return h * 3600 + m * 60 + s
                return float(time_str)
            
            # Apply conversion to start_time and end_time columns
            transcript_df['start_sec'] = transcript_df['token_start_time'].apply(time_to_seconds)
            transcript_df['end_sec'] = transcript_df['token_end_time'].apply(time_to_seconds)
            
            # Filter utterances in this time range
            filtered_df = transcript_df[
                ((transcript_df['start_sec'] >= start_time) & (transcript_df['start_sec'] < end_time)) |
                ((transcript_df['end_sec'] > start_time) & (transcript_df['end_sec'] <= end_time)) |
                ((transcript_df['start_sec'] <= start_time) & (transcript_df['end_sec'] >= end_time))
            ]
            
            if filtered_df.empty:
                transcript_text = ""
            else:   
                # Format transcript
                unique_utterances = filtered_df.drop_duplicates(subset=['utterance_id'])
                transcript_text = "\n".join([
                    f"{row.get('speaker', 'Unknown')}: {row.get('utterance', '')}"
                    for _, row in unique_utterances.iterrows()
                    if pd.notna(row.get('speaker', '')) and pd.notna(row.get('utterance', ''))
                ])
            
            # Add to results
            selected_chunks.append({
                'chunk_path': chunk_file,
                'chunk_number': chunk_id,
                'video_id': videostem,
                'chunk_start_time': start_time,
                'chunk_end_time': end_time,
                'transcript': transcript_text
            })
            #selected_chunk_ids.add(unique_chunk_id)
            
        except Exception as e:
            print(f"Error processing {transcript_path}: {e}")
    
    # Save results
    pd.DataFrame(selected_chunks).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(selected_chunks)} chunk transcripts to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
