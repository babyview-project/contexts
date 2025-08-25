import argparse
import os
from video_utils import assign_devices_to_ranks
from pathlib import Path
output_folder = "/ccn2/dataset/babyview/outputs_20250312"
frames_path_1k = f"{output_folder}/1000_random_frames.txt"
frames_path_10k = f"{output_folder}/10000_random_frames.txt"

def run_parallel_predictions(full_output_path, session_name, args):
    rank_device_dict = assign_devices_to_ranks(args.device_ids, args.num_parallel)
    # Create a new tmux session and split into the required number of panes
    os.system(f"tmux new-session -d -s {session_name}")
    for i in range(1, args.num_parallel):
        os.system(f"tmux split-window -t {session_name} -h")
        os.system(f"tmux select-layout -t {session_name} tiled")
    # Send the command to each pane
    for rank_id in rank_device_dict.keys():
        device_ids = rank_device_dict[rank_id]
        # for example if 1,2 then splitting at comma gives us length 2 which is the number of devices being passed in
        num_devices = len(device_ids.split(","))
        command = (f"source ~/miniconda3/bin/activate;conda activate activitycap;export VLLM_USE_V1=0;export VLLM_LOGGING_LEVEL=ERROR;export CUDA_VISIBLE_DEVICES={device_ids};"
               f"python image_locations_probs.py  --source {args.input_frames} --output {full_output_path} --prompting_batch {args.prompting_batch} " 
               f"--rank_id {rank_id} --num_parallel {args.num_parallel} --num_devices {num_devices}"
               f"{'--overwrite' if args.overwrite else ''}")
        os.system(f"tmux send-keys -t {session_name}.{rank_id} '{command}' Enter")
    print(f"Started {args.num_parallel} parallel processes in tmux session {session_name}")
    print(f"Use 'tmux attach -t {session_name}' to view progress")
    
def main():
    parser = argparse.ArgumentParser(description="Process frames using YOLOE.")
    parser.add_argument("--device_ids", type=str, default="[0]", help="List of GPU device IDs to use.")
    parser.add_argument("--num_parallel", type=int, default=1, help="Number of parallel processes.")
    parser.add_argument("--input_frames", type=str, default="/ccn2/dataset/babyview/outputs_20250312/sampled_frames", help="Text file or file path with the list of frames to be processed")
    parser.add_argument("--output_path", type=str, default=f"/ccn2/dataset/babyview/outputs_20250312/locations", help="Path to store outputs at")
    parser.add_argument("--prompting_batch", type=int, default=32, help="How big each batch size is when prompting the model")
    session_name_with_random_suffix = f"locations_predict_{os.urandom(4).hex()}"
    session_name = session_name_with_random_suffix
    parser.add_argument(
        "--overwrite",
        action='store_true',
        default=False,
        help="Whether to overwrite existing saved data"
    )
    args = parser.parse_args()
    full_output_path = args.output_path
    if args.input_frames == "10k":
        full_output_path = Path(f'{full_output_path}_10k')
        args.input_frames = frames_path_10k
    elif args.input_frames == "1k":
        full_output_path = Path(f'{full_output_path}_1k')
        args.input_frames = frames_path_1k
    else:
        run_parallel_predictions(full_output_path, session_name, args)

if __name__ == "__main__":
    main()