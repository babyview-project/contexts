import pandas as pd
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
from pathlib import Path

def analyze_location_changes(csv_file_path):
    """
    Step 1: Analyze CSV file to find location changes between chunks
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Sort by video name and chunk number to ensure proper order
    df = df.sort_values(['superseded_gcp_name_feb25', 'chunk_num']).reset_index(drop=True)
    
    location_changes = []
    
    # Group by video to analyze each video separately
    for video_name in df['superseded_gcp_name_feb25'].unique():
        video_df = df[df['superseded_gcp_name_feb25'] == video_name].copy()
        video_df = video_df.sort_values('chunk_num').reset_index(drop=True)
        
        if len(video_df) < 2:
            continue
            
        # Track consecutive chunks for each location
        current_location = video_df.iloc[0]['locations_chunked']
        current_count = 1
        start_chunk = video_df.iloc[0]['chunk_num']
        
        for i in range(1, len(video_df)):
            row = video_df.iloc[i]
            
            if row['locations_chunked'] != current_location:
                # Location changed - record this transition
                prev_location = current_location
                prev_count = current_count
                prev_end_chunk = video_df.iloc[i-1]['chunk_num']
                
                new_location = row['locations_chunked']
                new_start_chunk = row['chunk_num']
                
                # Count consecutive chunks for the new location
                new_count = 1
                j = i + 1
                while j < len(video_df) and video_df.iloc[j]['locations_chunked'] == new_location:
                    new_count += 1
                    j += 1
                
                # Get video path (assuming it's the same for all chunks of a video)
                from_video_path = video_df[video_df['chunk_num'] == prev_end_chunk]['video_path'].values[0]
                to_video_path = video_df[video_df['chunk_num'] == new_start_chunk]['video_path'].values[0]

                # Only save if both segments are long enough
                if prev_count >= 5 and new_count >= 5:
                    location_changes.append({
                        'video_name': video_name,
                        'from_location': prev_location,
                        'to_location': new_location,
                        'from_chunks_count': prev_count,
                        'to_chunks_count': new_count,
                        'transition_chunk': new_start_chunk,
                        'from_start_chunk': start_chunk,
                        'from_end_chunk': prev_end_chunk,
                        'to_start_chunk': new_start_chunk,
                        'from_video_path': from_video_path,
                        'to_video_path': to_video_path
                    })

                
                # Update for next iteration
                current_location = new_location
                current_count = new_count
                start_chunk = new_start_chunk
            else:
                current_count += 1
    
    return pd.DataFrame(location_changes)

def add_caption_to_frame(frame, change_info, segment_type=None):
    """
    Add text overlay to a single frame using OpenCV, indicating location change.
    
    Parameters:
        frame (np.array): Frame image.
        change_info (dict): Info about the location change.
        segment_type (str): "from" or "to" to label the segment type.
    """
    frame_with_text = frame.copy()

    # Label for segment type
    segment_label = "BEFORE (FROM)" if segment_type == "from" else "AFTER (TO)"

    # Prepare text lines
    lines = [
        f"Video: {change_info['video_name'][:50]}",
        f"Segment: {segment_label}",
        f"From: {change_info['from_location']} ({change_info['from_chunks_count']} chunks)",
        f"To: {change_info['to_location']} ({change_info['to_chunks_count']} chunks)",
        f"Transition at chunk: {change_info['transition_chunk']}"
    ]

    # Text styling
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    line_height = 30
    margin = 10

    # Calculate text background
    max_width = 0
    total_height = len(lines) * line_height + 2 * margin
    for line in lines:
        (text_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, text_width)

    # Add semi-transparent background rectangle
    overlay = frame_with_text.copy()
    cv2.rectangle(overlay, (0, 0), (max_width + 2 * margin, total_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame_with_text, 0.3, 0, frame_with_text)

    # Draw text
    y_position = margin + 25
    for line in lines:
        # Shadow
        cv2.putText(frame_with_text, line, (margin, y_position), font, font_scale, (0, 0, 0), thickness + 2)
        # Foreground
        cv2.putText(frame_with_text, line, (margin, y_position), font, font_scale, (255, 255, 255), thickness)
        y_position += line_height

    return frame_with_text


def calculate_chunk_timing(chunk_num, chunks_per_second=2):
    """
    Calculate the approximate time in seconds for a given chunk number.
    
    Parameters:
        chunk_num (int): The chunk number.
        chunks_per_second (float): The number of chunks per second. Default is 2.
        
    Returns:
        float: Time in seconds corresponding to the chunk number.
    """
    if chunk_num is None or pd.isna(chunk_num):
        return 0.0
    return chunk_num / chunks_per_second

def extract_video_segment_with_overlay(video_path, change_info, duration=5.0, start_time=0, segment_type="from"):
    """
    Extract a segment from the video and add text overlay using OpenCV
    """
    try:
        # Open video with OpenCV for frame processing
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frame range
        start_frame = int(start_time * fps)
        end_frame = min(int((start_time + duration) * fps), total_frames)
        
        if start_frame >= total_frames:
            start_frame = 0
            end_frame = min(int(duration * fps), total_frames)
        
        # Create temporary video file with overlay
        temp_video_path = f"temp_overlay_{segment_type}_{os.path.basename(video_path)}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))
        
        # Process frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Add text overlay to frame
            frame_with_overlay = add_caption_to_frame(frame, change_info, segment_type)
            out.write(frame_with_overlay)
            current_frame += 1
        
        cap.release()
        out.release()
        
        # Load the temporary video with MoviePy
        video_clip = VideoFileClip(temp_video_path)
        
        # Clean up temporary file
        try:
            os.remove(temp_video_path)
        except:
            pass
        
        return video_clip
        
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None

def create_location_change_compilation(csv_file_path, output_video_path, output_csv_path, segment_duration=10):
    """
    Step 2: Create video compilation of location changes
    """
    # Step 1: Analyze location changes
    print("Analyzing location changes...")
    changes_df = analyze_location_changes(csv_file_path)
    
    if changes_df.empty:
        print("No location changes found in the data.")
        return
    
    print(f"Found {len(changes_df)} location changes")
    
    # Save the analysis results to CSV
    changes_df.to_csv(output_csv_path, index=False)
    print(f"Location changes data saved to: {output_csv_path}")
    
    # Step 2: Create video segments with overlays
    video_segments = []
    
    for idx, change in changes_df.iterrows():
        print(f"Processing change {idx + 1}/{len(changes_df)}: {change['video_name']}")
        
        # Calculate timing for chunks (assuming chunks are taken every 10 seconds)
        from_chunk_time = calculate_chunk_timing(change.get('from_end_chunk'), chunks_per_second=0.1)
        to_chunk_time = calculate_chunk_timing(change.get('to_start_chunk'), chunks_per_second=0.1)
        
        print(f"  From chunk time: {from_chunk_time}, To chunk time: {to_chunk_time}")
        
        # Extract "FROM" segment (showing the location before change)
        from_start_time = max(0, from_chunk_time - segment_duration/2)
        from_segment = extract_video_segment_with_overlay(
            change['from_video_path'], 
            change,
            duration=segment_duration,
            start_time=from_start_time,
            segment_type="from"
        )

        # TO segment
        to_segment = extract_video_segment_with_overlay(
            change['to_video_path'], 
            change,
            duration=segment_duration,
            start_time=to_chunk_time,
            segment_type="to"
        )
        
        # Add both segments if they were successfully created
        if from_segment is not None:
            video_segments.append(from_segment)
            print(f"  Added FROM segment")
        else:
            print(f"  Failed to create FROM segment")
            
        if to_segment is not None:
            video_segments.append(to_segment)
            print(f"  Added TO segment")
        else:
            print(f"  Failed to create TO segment")
    
    if not video_segments:
        print("No valid video segments could be created.")
        return
    
    # Step 3: Concatenate all segments
    print("Creating final compilation video...")
    final_video = concatenate_videoclips(video_segments, method="compose")
    
    # Write the final video
    final_video.write_videofile(
        output_video_path,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )
    
    print(f"Compilation video saved to: {output_video_path}")
    
    # Clean up
    final_video.close()
    for segment in video_segments:
        segment.close()

# Example usage
if __name__ == "__main__":
    # Configuration
    csv_file_path = "all_chunks.csv"  # Replace with your CSV file path
    output_video_path = "location_changes_compilation.mp4"
    output_csv_path = "location_changes_analysis.csv"
    segment_duration = 10  # Duration of each video segment in seconds
    
    # Make sure the CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"CSV file not found: {csv_file_path}")
        print("Please update the csv_file_path variable with the correct path to your CSV file.")
    else:
        # Run the analysis and video creation
        create_location_change_compilation(
            csv_file_path=csv_file_path,
            output_video_path=output_video_path,
            output_csv_path=output_csv_path,
            segment_duration=segment_duration
        )