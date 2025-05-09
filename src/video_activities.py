from tqdm import tqdm
import pandas as pd
from models.video.videochat_flash import VideoFlash
from models.image.internvl import InternVL
locations = ["bathroom", "bedroom", "car", "closet", "garage", "living room", "hallway", "outside", "garage", "kitchen", "deck"]
activities = ["being held", "eating", "drinking", "playing with toy", "getting changed", "crawling", "crying", "exploring", "cooking", "cleaning", "gardening", "watching tv", "driving", "reading"]
video_generator = VideoFlash()
chatter = InternVL()
chunks = pd.read_csv("selected_chunk_transcripts.csv")
df = []
for i, (video_path, transcript) in tqdm(enumerate(zip(chunks["chunk_path"], chunks["transcript"])), desc="Getting video descriptions"):
    location, _ = video_generator.caption_video(video_path,f"Answer with one word what location this video, taken with a camera attached to the head of a child, is in from the following options: {", ".join(locations)}")
    activity, _ = video_generator.caption_video(video_path,f"Answer with one word what activity is going on in this video, taken with a camera attached to the head of a child, from the following options: {", ".join(activities)}")
    activity_transcript, _ = video_generator.caption_video(video_path,f"Given that the transcript of this video is: {transcript}, answer with one word what activity is going on in this video, taken with a camera attached to the head of a child, from the following: {", ".join(activities)}")
    explain_video, _ = video_generator.caption_video(video_path, f"Detail this video.")
    vid_transcript_lm = chatter.ask_question(f"Video caption: {explain_video}\nTranscript: {transcript}\nAnswer with one phrase what activity is going on in this video, taken with a camera attached to the head of a child, from the following options: {", ".join(activities)}")
    df.append({
        "video_path": video_path,
        "location": location,
        "activity": activity,
        "transcript": transcript,
        "activity_transcript": activity_transcript,
        "vid_transcript_lm": vid_transcript_lm,
        "video_description": explain_video
    })
    pd.DataFrame(df).to_csv("video_activities_locations.csv", index=False)
