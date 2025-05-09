import os
import math
from pathlib import Path
import subprocess
import numpy as np
from decord import VideoReader, cpu
import pandas as pd

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


def update_csv_with_batch_results(current_batch, vid_paths, df_name="video_activities_locations_probs.csv"):
    """
    Update the main CSV file with results from the current batch
    
    Args:
        current_batch: DataFrame containing updated data for the current batch
        vid_paths: List of video paths processed in the current batch
    """
    try:
        # Read the original CSV file
        original_df = pd.read_csv(df_name)
        
        # Make sure all required columns exist in the original DataFrame
        new_columns = ["text_options", "text_probs"] #"samples", "sample_probs"
        for col in new_columns:
            if col not in original_df.columns:
                original_df[col] = ""
        
        # Filter the current batch to include only processed video paths
        updated_batch = current_batch[current_batch["video_path"].isin(vid_paths)].copy()
        
        # Create an index based on video_path for efficient updates
        original_df.set_index("video_path", inplace=True)
        updated_batch.set_index("video_path", inplace=True)
        
        # Update the original DataFrame with values from the current batch
        for idx in updated_batch.index:
            if idx in original_df.index:
                for col in new_columns:
                    original_df.loc[idx, col] = updated_batch.loc[idx, col]
        
        # Reset index before saving
        original_df.reset_index(inplace=True)
        
        # Save the updated DataFrame
        original_df.to_csv(df_name, index=False)
        
    except Exception as e:
        print(f"Error updating CSV: {e}")
