from video_utils import assign_devices_to_ranks
import os
import argparse

def run_parallel_predictions(session_name, args):
    rank_device_dict = assign_devices_to_ranks(args.device_ids, args.num_parallel)
    # Create a new tmux session and split into the required number of panes
    os.system(f"tmux new-session -d -s {session_name}")
    for i in range(1, args.num_parallel):
        os.system(f"tmux split-window -t {session_name} -h")
        os.system(f"tmux select-layout -t {session_name} tiled")
    # Send the command to each pane
    print(rank_device_dict)
    for rank_id in rank_device_dict.keys():
        device_ids = rank_device_dict[rank_id]
        command = (f"conda activate activitycap;export CUDA_VISIBLE_DEVICES={device_ids};"
               f"python -m chunking.chunker --input {args.input} --output_dir {args.output_dir} " 
               f"--rank_id {rank_id} --num_parallel {args.num_parallel}")
        os.system(f"tmux send-keys -t {session_name}.{rank_id} '{command}' Enter")
    print(f"Started {args.num_parallel} parallel processes in tmux session {session_name}")
    print(f"Use 'tmux attach -t {session_name}' to view progress")

def main():
    parser = argparse.ArgumentParser(description="Chunk videos into 1 minute segments")
    parser.add_argument("--device_ids", type=str, default="[0]", help="List of GPU device IDs to use.")
    parser.add_argument("--num_parallel", type=int, default=1, help="Number of parallel processes.")
    parser.add_argument("--input", type=str, default="all_video_paths.csv", help="Text file or file path with the list of frames to be processed")
    parser.add_argument("--output_dir", type=str, default=f"/ccn2/dataset/babyview/outputs_20250312/locations", help="Path to store outputs at")
    session_name_with_random_suffix = f"videos_chunk_{os.urandom(4).hex()}"
    session_name = session_name_with_random_suffix
    args = parser.parse_args()
    run_parallel_predictions(session_name, args)

if __name__ == "__main__":
    main()