
"""
Create video embeddings for videos in a directory using a pre-trained V-JEPA 2 model.

conda activate babyview-pose
cd /ccn2/u/khaiaw/Code/babyview-pose/contexts/video-embedding/


    --input_video_dir /ccn2/dataset/kinetics400/Kinetics400/k400/train/ \
    --out_dir /ccn2a/dataset/babyview/2025.2/outputs/video_embeddings/kinetics_train/ \
        
    --input_video_dir /ccn2/u/wanhee/datasets/gen2/egoexo4d/ns_process_mp4/video \
    --out_dir /ccn2a/dataset/babyview/2025.2/outputs/video_embeddings/egoexo4d/ \

    --input_video_dir /ccn2/dataset/SAYCam/ \
    --out_dir /ccn2a/dataset/babyview/2025.2/outputs/video_embeddings/SAYCam/ \

    --input_video_dir /ccn2a/dataset/babyview/2025.2/split_10s_clips_256p/ \
    --out_dir /ccn2a/dataset/babyview/2025.2/outputs/video_embeddings/babyview/ \

    --input_video_dir /ccn2/dataset/ego4D/v1/chunked_resized/ \
    --out_dir /ccn2a/dataset/babyview/2025.2/outputs/video_embeddings/ego4D/ \
        
    --input_video_dir /ccn2/dataset/Moments/Moments_in_Time_Raw/training \
    --out_dir /ccn2a/dataset/babyview/2025.2/outputs/video_embeddings/Moments_in_Time_Raw_training/ \

    --input_video_dir /data2/klemenk/ssv2/ \
    --out_dir /ccn2a/dataset/babyview/2025.2/outputs/video_embeddings/ssv2/ \

    --input_video_dir /ccn2a/dataset/physion/ \
    --out_dir /ccn2a/dataset/babyview/2025.2/outputs/video_embeddings/physion/ \
        
export CUDA_VISIBLE_DEVICES=1
python hf_create_embeddings.py \
    --input_video_dir /ccn2/dataset/ego4D/v1/chunked_resized/ \
    --out_dir /ccn2a/dataset/babyview/2025.2/outputs/video_embeddings/ego4D/ \
    --model_name facebook/vjepa2-vitl-fpc64-256 \
    --num_processes 4 \
    --debug \

"""

import os
import copy
import argparse
import sys
import numpy as np
import yaml
import torch
import torch.nn.functional as F
import glob
import decord
from decord import VideoReader, cpu

import torch
from torchcodec.decoders import VideoDecoder
import numpy as np
from transformers import AutoVideoProcessor, AutoModel
import ray

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name of the pre-trained model')
    parser.add_argument('--input_video_dir', type=str, default='/ccn2a/dataset/babyview/2025.2/split_10s_clips_256p/', help='Directory of input images')
    parser.add_argument('--out_dir', type=str, default='/ccn2/u/khaiaw/Code/counterfactual_benchmark/model_predictions/ObjectReasoning/vjepa2', help='Directory to save output images')
    parser.add_argument('--debug', action='store_true', help='If true, run in debug mode (process only 1 video)')
    parser.add_argument('--num_processes', type=int, default=16, help='Number of processes to use')
    return parser.parse_args()

@torch.no_grad()
def create_video_embedding(args, video_url, processor, model):

    video_id = os.path.basename(video_url).replace('.mp4', '')
    out_path = os.path.join(args.out_dir, f'{video_id}.npy')
    if os.path.exists(out_path):
        return

    # Get video and its metadata
    vr = VideoDecoder(video_url)
    fps = vr.metadata.average_fps
    num_total_frames = vr.metadata.num_frames
    duration_seconds = vr.metadata.duration_seconds
    if duration_seconds < 2:
        return None
    if duration_seconds > 12: # if the clip is longer than 12s, limit to first 10 seconds, otherwise the frames will be spaced too far apart
        num_total_frames = int(fps * 10) 

    # Sample frames linearly spaced throughout the video
    n_frames_to_sample = 64
    frame_idx = np.linspace(0, num_total_frames-1, n_frames_to_sample).astype(int)
    
    video = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
    video = processor(video, return_tensors="pt").to(model.device)
    outputs = model(**video)

    # V-JEPA 2 encoder outputs, same as calling `model.get_vision_features()`
    encoder_outputs = outputs.last_hidden_state # torch.Size([1, 8192, 1024])
    
    # 1D vector representation of the video
    video_embedding = encoder_outputs.squeeze(0).mean(dim=0) # torch.Size([1024])
    video_embedding = video_embedding.cpu().numpy().astype(np.float16)  # Move to CPU and convert to numpy, (1024,)

    # Save the video embedding
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, video_embedding)

def get_model_and_processor(args):
    processor = AutoVideoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(
        args.model_name,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    return processor, model

@ray.remote(num_gpus=0.25)
def process_list_of_videos(args, video_files):
    processor, model = get_model_and_processor(args)
    
    for video_file in video_files:
        try:
            create_video_embedding(args, video_file, processor, model)
        except Exception as e:
            print(f'Error processing {video_file}: {e}')
    

if __name__ == "__main__":
    args = get_args()
    args.out_dir = os.path.join(args.out_dir, args.model_name.replace('/', '_'))

    # video_files = [
    #     os.path.join(args.input_video_dir, file)
    #     for file in os.listdir(args.input_video_dir)
    #     if file.endswith('.mp4')
    # ]
    video_files = glob.glob(os.path.join(args.input_video_dir, '**/*.mp4'), recursive=True)
    print(f'Found {len(video_files)} videos in {args.input_video_dir}')
    np.random.shuffle(video_files)
    if args.debug:
        processor, model = get_model_and_processor(args)
        video_files = video_files[:3]
        for video_file in video_files:
            create_video_embedding(args, video_file, processor=processor, model=model)
    else:
        chunk_size = len(video_files) // args.num_processes + 1
        video_chunks = [video_files[i:i + chunk_size] for i in range(0, len(video_files), chunk_size)]
        tasks = [process_list_of_videos.remote(args, chunk) for chunk in video_chunks]
        
        ray.init(ignore_reinit_error=True)
        ray.get(tasks)
        ray.shutdown()
        
    