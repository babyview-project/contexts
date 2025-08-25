from tqdm import tqdm
import pandas as pd
from models.video.videochat_flash import VideoFlash
from constants import get_activities, get_locations
import ray
import argparse

locations = get_locations()
activities = get_activities()
df = []
num_processes=1

@ray.remote(num_gpus=1)
def run_model(chunks, output):
    video_generator = VideoFlash()
    for i, (video_path, transcript) in tqdm(enumerate(zip(chunks["chunk_path"], chunks["transcript"])), desc="Getting video descriptions", total=len(chunks["transcript"])):
        activity_transcript, _ = video_generator.caption_video(video_path,f"This is a video from the point-of-view of a head-mounted camera on child. The transcript of this video is: {transcript}. Respond strictly only in this format with both keys and values: Activity: <{"/".join(activities)}>")
        location_transcript, _ = video_generator.caption_video(video_path,f"This is a video from the point-of-view of a head-mounted camera on child. The transcript of this video is: {transcript}. Respond strictly only in this format with both keys and values: Location: <{"/".join(locations)}>")
        explain_video, _ = video_generator.caption_video(video_path, f"Describe this video in detail.")
        lines = [activity_transcript, location_transcript]
        lines = [line.strip() for line in lines if line.strip()]
        lines = [line.replace('<', '').replace('>', '') for line in lines]
        location = ""
        activity = ""
        for key in ["Location", "Activity"]:
            for line in lines:
                line_split = line.split(":")
                answer = line_split[-1].strip()
                if len(line_split) <= 1 or len(answer) < 2:
                    print(f"Answer for {key} too short, skipping")
                    continue
                if line.startswith(key):
                    if key == "Location":
                        location = answer
                    elif key == "Activity":
                        activity = answer
        df.append({
            "video_path": video_path,
            "location": location,
            "transcript": transcript,
            "activity": activity,
            "video_description": explain_video
        })
        pd.DataFrame(df).to_csv(output, index=False)

def main():
    parser = argparse.ArgumentParser(description='Get video descriptions, locations, and activities from VQA model')
    parser.add_argument('--csv_path', default="selected_chunk_transcripts_0530.csv", help='Path to input CSV file')
    parser.add_argument('--output', '-o', 
                      help='Path to output CSV file',
                      default="video_activities_locations.csv")
    
    args = parser.parse_args()
    ray.init(num_cpus = num_processes)
    all_chunks = pd.read_csv(args.csv_path)
    chunk_size = len(all_chunks) // num_processes + (1 if len(all_chunks) % num_processes else 0)
    video_chunks = [all_chunks[i:i+chunk_size] for i in range(0, len(all_chunks), chunk_size)]
    print(video_chunks)
    print(len(video_chunks))
    futures = [run_model.remote(chunk, f"vid_activities_{i}.csv") for i, chunk in enumerate(video_chunks)]
    ray.get(futures)  
    
parser = argparse.ArgumentParser(description="Grab video descriptions from video clips and transcripts")
if __name__ == "__main__":
    main()