import av
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModelForCausalLM
from dotenv import load_dotenv
import os
import shutil
import csv
import pandas as pd

load_dotenv()

processor = AutoProcessor.from_pretrained("microsoft/git-base-vatex")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vatex")
project_dir = os.environ.get("PROJECT_PATH")

# set seed for reproducability
np.random.seed(52)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def sample_frame_indices_all(clip_len, frame_sample_rate, seg_len):
    '''
    Create a list of lists of frame indices ensuring the entire video is included.
    Args:
        clip_len (int): Total number of frames to sample per segment.
        frame_sample_rate (int): Sample every n-th frame.
        seg_len (int): Total number of frames in the video.
    Returns:
        indices_list (List[List[int]]): List of lists of sampled frame indices for each segment.
    '''
    indices_list = []
    converted_len = clip_len * frame_sample_rate
    
    # Divide the video into segments of `converted_len` frames each
    for start_idx in range(0, seg_len, converted_len):
        end_idx = min(start_idx + converted_len, seg_len)
        num_frames_in_segment = (end_idx - start_idx) // frame_sample_rate
        sampled_indices = np.arange(start_idx, end_idx, frame_sample_rate)[:clip_len]
        
        # If not enough frames for the clip, pad with the last frame of the segment
        if len(sampled_indices) < clip_len:
            padding = [sampled_indices[-1]] * (clip_len - len(sampled_indices))
            sampled_indices = np.concatenate([sampled_indices, padding])
        
        indices_list.append(sampled_indices.astype(int).tolist())
    
    return indices_list

def append_to_csv(file_path, frame_sample_rate, timestamps, frame_rate, captions, video_name):
    '''
    Append data to a CSV file or create the file if it doesn't exist.
    Args:
        file_path (str): Path to the CSV file.
        frame_sample_rate (int): Frame sampling rate.
        timestamps (Tuple[float, float]): First and last timestamps.
        frame_rate (float): Frame rate of the video.
        captions (List[str]): Generated captions.
    '''
    # Create a dictionary for the new data
    data = {
        'frame_sample_rate': [frame_sample_rate],
        'first_timestamp': [round(timestamps[0],2)],
        'last_timestamp': [round(timestamps[1],2)],
        'frame_rate': [round(frame_rate,2)],
        'captions': [captions],
        'video_name': [video_name]  # Combine captions into a single string
    }
    
    # Convert to DataFrame
    new_df = pd.DataFrame(data)
    
    # Check if the file exists
    file_exists = os.path.isfile(file_path)
    
    # Append to the file, writing the header only if the file does not exist
    new_df.to_csv(file_path, mode='a', header=not file_exists, index=False)

def get_first_and_last_timestamps(indices, frame_rate):
    '''
    Get the first and last timestamps for a segment of sampled frames.
    Args:
        indices (List[List[int]]): Sampled frame indices for the current segment.
        frame_rate (float): Frame rate of the video (frames per second).
    Returns:
        timestamps_list ([Tuple[float, float]]): Tuples (first_timestamp, last_timestamp) for the current segment.
    '''
    first_timestamp = indices[0] / frame_rate
    last_timestamp = indices[-1] / frame_rate
    return [first_timestamp, last_timestamp]

# load video
def get_all_captions(file_path, csv_file_path, input_frame_sample_rate=4):
    container = av.open(file_path)
    total_frames = container.streams.video[0].frames
    frame_rate = float(container.streams.video[0].average_rate)
    print(f"Total frames: {total_frames}, Frame rate: {frame_rate}")

    # sample frames
    num_frames = model.config.num_image_with_embedding
    indices_set = sample_frame_indices_all(
        clip_len=num_frames, frame_sample_rate=input_frame_sample_rate, seg_len=total_frames
    )
    
    for indices in indices_set:
        frames = read_video_pyav(container, indices)
        pixel_values = processor(images=list(frames), return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        [first_timestamp, last_timestamp] = get_first_and_last_timestamps(indices, frame_rate)
        # Append to CSV
        append_to_csv(
        file_path=csv_file_path,
        frame_sample_rate=input_frame_sample_rate,
        timestamps=(first_timestamp, last_timestamp),
        frame_rate=frame_rate,
        captions=processor.batch_decode(generated_ids, skip_special_tokens=True),
        video_name=os.path.splitext(os.path.basename(file_path))[0],
        )

get_all_captions(os.path.join(project_dir, "data", "babyview-example-3.mp4"), os.path.join(project_dir, "data", "git_captions.csv"))
get_all_captions(os.path.join(project_dir, "data", "babyview-example-3.mp4"), os.path.join(project_dir, "data", "git_captions.csv"), input_frame_sample_rate=10)