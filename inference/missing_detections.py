import os
import pandas as pd
import torch
import re
from tqdm import tqdm
from models.video.videollama_constrained import VideoLLaMA
from models.video.vqa import _set_seed
from constants import get_activities, get_locations, get_bv_main_ids
import argparse

"""
Script to fix missing Location detections in videollama3_constrained CSV files
Usage: python fix_missing_locations.py --csv_dir path/to/videollama3_constrained
"""

_set_seed(42)

prompt_key_values = {
    "Location": get_locations(),
    "Activity": get_activities(),
    "Video description": None
}

def create_question():
    question = "This video is recorded from the point-of-view of a child, with a camera mounted on the child's head. Respond strictly only in this format with both keys and values: "
    for key, values in prompt_key_values.items():
        if values is None:
            prompt_value = "..."
        else:
            prompt_value =  f"<{'/'.join(values)}>"
        question += f"{key}: {prompt_value} || "
    return question

def find_csv_files_with_missing_locations(csv_dir):
    """Find all CSV files and identify which ones have missing Location values"""
    csv_files = []
    missing_stats = {}
    
    print("Scanning CSV files for missing Location values...")
    for filename in tqdm(os.listdir(csv_dir), desc="Scanning CSV files"):
        if filename.endswith('.csv'):
            filepath = os.path.join(csv_dir, filename)
            try:
                df = pd.read_csv(filepath)
                if 'Location' in df.columns:
                    # Find rows with missing or empty Location values
                    missing_mask = df['Location'].isna() | (df['Location'] == '') | (df['Location'].str.strip() == '')
                    missing_count = missing_mask.sum()
                    
                    if missing_count > 0:
                        csv_files.append(filepath)
                        missing_stats[filepath] = {
                            'total_rows': len(df),
                            'missing_count': missing_count,
                            'missing_indices': df[missing_mask].index.tolist()
                        }
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return csv_files, missing_stats

def get_new_response_for_video(prompter, video_path, question, max_retries=5):
    """Get a new response for a specific video"""
    response = None
    response_dict = None
    try_count = 0
    
    while response is None or response_dict is None:
        try_count += 1
        if try_count > max_retries:
            print(f"Failed to get response after {max_retries} tries for: {video_path}")
            return None
        
        try:
            response = prompter.get_response_with_constraints(video_path, question, prompt_key_values)
            print(response)
            response_dict = prompter.convert_model_response_to_dict(response, list(prompt_key_values.keys()), prompt_key_values)
            
            # Check if Location is still missing
            if response_dict and (not response_dict.get('Location') or response_dict.get('Location').strip() == ''):
                response_dict = None  # Force retry
                continue
                
        except Exception as e:
            print(f"Error getting response for {video_path} (attempt {try_count}): {e}")
            response_dict = None
    
    return response_dict

def process_csv_file(filepath, missing_info, prompter, question):
    """Process a single CSV file and fix missing Location values"""
    print(f"\nProcessing {os.path.basename(filepath)}")
    print(f"  Total rows: {missing_info['total_rows']}, Missing locations: {missing_info['missing_count']}")
    
    # Load the CSV
    df = pd.read_csv(filepath)
    missing_indices = missing_info['missing_indices']
    
    updated_count = 0
    failed_count = 0
    
    # Process each row with missing Location
    for idx in tqdm(missing_indices, desc=f"Fixing {os.path.basename(filepath)}", leave=False):
        row = df.iloc[idx]
        
        # Try to find the video file
        video_path = None
        if 'video_path' in row and pd.notna(row['video_path']) and os.path.exists(row['video_path']):
            video_path = row['video_path']
        elif 'superseded_gcp_name_feb25' in row and pd.notna(row['superseded_gcp_name_feb25']):
            video_path = row['superseded_gcp_name_feb25']
        elif 'video_id' in row and pd.notna(row['video_id']):
            # Try to construct path from video_id
            video_id = row['video_id']
            # This might need adjustment based on your directory structure
            potential_paths = [
                os.path.join('/ccn2/dataset/babyview/unzip_2025_10s_videos_256p/', video_id),
                # Add other potential path patterns here
            ]
            for path in potential_paths:
                if os.path.exists(path):
                    video_path = path
                    break
        
        if not video_path or not os.path.exists(video_path):
            print(f"  Could not find video file for row {idx}, skipping...")
            failed_count += 1
            continue
        
        # Get new response
        new_response = get_new_response_for_video(prompter, video_path, question)
        
        if new_response:
            # Update the row with new values
            for key in prompt_key_values.keys():
                if key in new_response and key in df.columns:
                    df.at[idx, key] = new_response[key]
            updated_count += 1
            print(f"  Updated row {idx}: Location = '{new_response.get('Location', 'N/A')}'")
        else:
            failed_count += 1
            print(f"  Failed to get new response for row {idx}")
    
    # Save the updated CSV
    if updated_count > 0:
        df.to_csv(filepath, index=False)
        print(f"  Saved {updated_count} updates to {filepath}")
    
    return updated_count, failed_count

def main():
    parser = argparse.ArgumentParser(description='Fix missing Location detections in CSV files')
    parser.add_argument('--csv_dir', type=str, default="../predictions/videollama3_constrained", 
                       help='Directory containing CSV files to process')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of CSV files to process (for testing)')
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_dir):
        print(f"Error: Directory {args.csv_dir} does not exist")
        return
    
    # Find CSV files with missing locations
    csv_files, missing_stats = find_csv_files_with_missing_locations(args.csv_dir)
    
    if not csv_files:
        print("No CSV files with missing Location values found!")
        return
    
    print(f"\nFound {len(csv_files)} CSV files with missing Location values:")
    total_missing = sum(stats['missing_count'] for stats in missing_stats.values())
    print(f"Total missing locations to fix: {total_missing}")
    
    # Limit files if specified
    if args.max_files:
        csv_files = csv_files[:args.max_files]
        print(f"Processing first {len(csv_files)} files only")
    
    # Initialize the model
    print("\nInitializing VideoLLaMA model...")
    prompter = VideoLLaMA(use_constrained_decoding=True)
    question = create_question()
    
    # Process each CSV file
    total_updated = 0
    total_failed = 0
    
    print(f"\nProcessing {len(csv_files)} CSV files...")
    for filepath in tqdm(csv_files, desc="Processing CSV files"):
        try:
            updated, failed = process_csv_file(filepath, missing_stats[filepath], prompter, question)
            total_updated += updated
            total_failed += failed
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            total_failed += missing_stats[filepath]['missing_count']
    
    print(f"\n=== SUMMARY ===")
    print(f"Total locations updated: {total_updated}")
    print(f"Total failures: {total_failed}")
    print(f"Files processed: {len(csv_files)}")
    print("Processing complete!")

if __name__ == "__main__":
    main()