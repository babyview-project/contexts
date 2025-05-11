import os
import math
from pathlib import Path
import subprocess
import numpy as np
from decord import VideoReader, cpu
import pandas as pd
import re

def split_video_fast(video_path, chunk_size_sec, output_dir, keep_audio=True):
    """
    Fast video splitting with re-encoding for Decord compatibility.
    
    Args:
        video_path (str): Path to input video.
        chunk_size_sec (int): Duration of each chunk in seconds.
        output_dir (str): Directory to save video chunks.
        keep_audio (bool): Whether to keep audio in chunks.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get video duration
    duration_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', video_path
    ]
    duration = float(subprocess.check_output(duration_cmd).decode().strip())
    num_chunks = math.ceil(duration / chunk_size_sec)

    for i in range(num_chunks):
        start = i * chunk_size_sec
        output_filename = f"chunk{i:03d}.mp4"  # zero-padded to 3 digits
        output_path = os.path.join(output_dir, output_filename)

        cmd = [
            'ffmpeg', '-y',
            '-loglevel', 'error',
            '-ss', str(start),
            '-t', str(chunk_size_sec),
            '-i', video_path,
            '-c:v', 'h264_nvenc',
            '-preset', 'fast',
            '-crf', '23',
            '-threads', '0'  # use all CPU cores
        ]

        if keep_audio:
            cmd += ['-c:a', 'aac', '-b:a', '96k']
        else:
            cmd += ['-an']  # no audio

        cmd.append(output_path)

        try:
            subprocess.run(cmd, check=True)
            #print(f"Chunk {i+1:03d}/{num_chunks:03d} created: {output_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error in chunk {i+1}: {e.stderr.decode() if e.stderr else e}")

def split_video(video_path, chunk_size_sec, output_dir, force_keyframes=True):
    """
    Split a video into chunks of specified duration.
    
    Args:
        video_path: Path to the source video file
        chunk_size_sec: Length of each chunk in seconds
        output_dir: Directory to save the chunks
        force_keyframes: If True, re-encode to ensure clean splits; if False, use stream copy
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)    
    
    # Get video duration using ffprobe
    duration_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', video_path
    ]
    duration = float(subprocess.check_output(duration_cmd).decode('utf-8').strip())
    
    # Calculate number of chunks
    num_chunks = math.ceil(duration / chunk_size_sec)
    
    # Get source video info for codec selection
    codec_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name', '-of', 
        'default=noprint_wrappers=1:nokey=1', video_path
    ]
    video_codec = subprocess.check_output(codec_cmd).decode('utf-8').strip()
    
    for chunk_idx in range(num_chunks):
        # Calculate start and end times for this chunk
        start_time = chunk_idx * chunk_size_sec
        
        # Create output filename
        output_filename = f"chunk{chunk_idx+1}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Base command
        cmd = [
            'ffmpeg', '-y', 
            '-loglevel', 'error',
            '-i', video_path,
            '-ss', str(start_time),
            '-t', str(chunk_size_sec),
        ]
        
        # Select encoding method
        if force_keyframes:
            # Re-encode for clean splits (slower but more reliable)
            # Choose a good codec based on source
            if video_codec in ["h264", "libx264"]:
                cmd.extend(['-c:v', 'libx264', '-preset', 'fast', '-crf', '22'])
            elif video_codec in ["hevc", "h265", "libx265"]:
                cmd.extend(['-c:v', 'libx265', '-preset', 'fast', '-crf', '28'])
            else:
                cmd.extend(['-c:v', 'libx264', '-preset', 'fast', '-crf', '22'])  # Default to h264
            
            # Copy audio stream without re-encoding
            cmd.extend(['-c:a', 'copy'])
        else:
            # Fast method: just copy streams (may have issues at chunk boundaries)
            cmd.extend(['-c', 'copy'])
        
        # Add output path
        cmd.append(output_path)
        
        # Run FFmpeg command
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"Created chunk {chunk_idx+1}/{num_chunks}: {output_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating chunk {chunk_idx+1}: {e.stderr.decode() if e.stderr else 'Unknown error'}")
    
    print(f"Video split into {num_chunks} chunks successfully.")

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_idx = [i for i in frame_idx if i < total_frame_num]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

def batcher(df, batch_num, total_batches):
    total_paths = len(df)
    batch_size = total_paths // total_batches
    remainder = total_paths % total_batches

    if batch_num <= remainder:
        # First 'remainder' batches get one extra item
        start = (batch_size + 1) * (batch_num - 1)
        end = start + (batch_size + 1)
    else:
        # Remaining batches get standard batch_size
        start = (batch_size + 1) * remainder + batch_size * (batch_num - remainder - 1)
        end = start + batch_size
    return df[start:end]

def softmax(logits):
    logits = np.array(logits)
    # Subtract max for numerical stability
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)


def update_csv_with_batch_results(current_batch, vid_paths, df_name="video_activities_locations_probs.csv", keypath="image_path"):
    """
    Update the main CSV file with results from the current batch
    
    Args:
        current_batch: DataFrame containing updated data for the current batch
        vid_paths: List of video/image paths processed in the current batch
    """
    try:
        updated_batch = current_batch[current_batch[keypath].isin(vid_paths)].copy()
        if os.path.exists(df_name):
            # Read the original CSV file
            original_df = pd.read_csv(df_name)

            # Merge the original and updated batch, prioritizing updated_batch values
            combined_df = pd.concat([original_df, updated_batch], ignore_index=True)

            # Drop duplicates based on keypath, keeping the last occurrence (i.e., from updated_batch)
            combined_df.drop_duplicates(subset=keypath, keep='last', inplace=True)

            # Save the combined DataFrame
            combined_df.to_csv(df_name, index=False)
        else:
            # If the file doesn't exist, create a new one from the current batch
            dir_name = os.path.dirname(df_name)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            updated_batch.to_csv(df_name, index=False)
    except Exception as e:
        print(f"Error updating CSV: {e}")

def assign_top_label_and_probs(generations, image_paths, label_type, choices_regex,current_batch=None):
    """
    Updates current_batch DataFrame with options, probabilities, and top choice for a given label type.
    If current_batch is empty, it will be created with appropriate columns.
    """
    rows_to_add = []

    for prompt_generation, image_path in zip(generations, image_paths):
        text_options_dict = {}

        for i, output in enumerate(prompt_generation.outputs):
            curr_text = output.text
            curr_prob = output.cumulative_logprob
            text_options_dict[curr_text] = curr_prob

            if i == 0:
                for logprob in output.logprobs[0].values():
                    if re.match(choices_regex, logprob.decoded_token):
                        text_options_dict[logprob.decoded_token] = logprob.logprob

        text_options_list = list(text_options_dict.keys())
        probs_list = [text_options_dict[opt] for opt in text_options_list]
        if not probs_list:
            continue

        probs_softmax = softmax(np.array(probs_list))
        filtered = [(opt, round(prob, 2)) for opt, prob in zip(text_options_list, probs_softmax) if prob >= 0.1]
        filtered_text_options, filtered_probs = zip(*filtered) if filtered else ([], [])

        row = {
            "image_path": image_path,
            f"{label_type}_options": ",".join(filtered_text_options),
            f"{label_type}_probs": ",".join(map(str, filtered_probs)),
            label_type: filtered_text_options[np.argmax(filtered_probs)] if filtered_probs else ""
        }
        rows_to_add.append(row)

    new_rows_df = pd.DataFrame(rows_to_add)

    if current_batch is None or current_batch.empty:
        return new_rows_df
    else:
        current_batch = current_batch.set_index("image_path")
        new_rows_df = new_rows_df.set_index("image_path")
        for col in new_rows_df.columns:
            current_batch.loc[new_rows_df.index, col] = new_rows_df[col]
        current_batch = current_batch.reset_index()
        return current_batch

def assign_devices_to_ranks(device_ids_str, num_parallel):
    """
    Assigns GPU devices to each parallel rank, supporting uneven splits.

    Args:
        device_ids_str (str): e.g. "[0,1,2,3,4]"
        num_parallel (int): Number of parallel runs (ranks)

    Returns:
        dict: {rank_id: "0,1"} for setting CUDA_VISIBLE_DEVICES
    """
    device_ids = [int(id.strip()) for id in device_ids_str.strip("[]").split(",")]
    num_devices = len(device_ids)

    devices_per_run = num_devices // num_parallel
    remaining_devices = num_devices % num_parallel

    rank_device_dict = {}
    start_idx = 0

    for i in range(num_parallel):
        end_idx = start_idx + devices_per_run
        if i == num_parallel - 1:
            end_idx += remaining_devices  # Add leftover devices to last rank
        assigned_devices = device_ids[start_idx:end_idx]
        rank_device_dict[i] = ",".join(map(str, assigned_devices))
        start_idx = end_idx

    return rank_device_dict

def largest_factor_less_than(m, n):
    for i in range(m, 0, -1): 
        if n % i == 0:
            return i
    return None 
