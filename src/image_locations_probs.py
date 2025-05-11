import re
from re import escape as regex_escape
from collections.abc import Sequence
import pandas as pd
import os
import math
from vllm.model_executor.guided_decoding.outlines_logits_processors import RegexLogitsProcessor
import argparse
from tqdm import tqdm
from dataclasses import asdict
from video_utils import batcher, softmax, update_csv_with_batch_results, assign_top_label_and_probs, largest_factor_less_than
from transformers import AutoTokenizer
import numpy as np
import PIL
# from vllm.sampling_params import SamplingParams
from vllm import LLM, SamplingParams,  EngineArgs
from vllm.sampling_params import GuidedDecodingParams
from typing import NamedTuple, Optional
from vllm.lora.request import LoRARequest
from pathlib import Path

activities = ['being held',  'eating',  'drinking',  'playing with toy',  'getting changed',  'crawling',  'crying',  'exploring',  'cooking',  'cleaning',  'gardening',  'watching tv',  'driving',  'reading',  'looking at device', 'dancing', 'music time', 'nothing', 'nursing']
locations = ["bathroom", "bedroom", "car", "closet", "garage", "living room", "hallway", "outside", "garage", "kitchen", "deck"]

# PROMPTS
base_location_prompt = f"Answer with one word what location this image is in from the following, taken with a camera attached to the head of a child: {", ".join(locations)}"
base_activity_prompt = f"Answer with one word what activity is going on in this image from the following, taken with a camera attached to the head of a child: {", ".join(activities)}"

# SAMPLING PARAMS
sampling_nums = 5
logprobs = 2
set_seed = 10

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None

def parse_args():
    parser = argparse.ArgumentParser(description="Find probabilities of locations and activities in images.")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the input image, directory, or text file with file paths"
    )
    parser.add_argument("--prompting_batch", type=int, default=50, help="How many prompts to pass into a single generate call")
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the annotated image",
        default="outputs_new.csv"
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        default=False,
        help="Whether to overwrite existing saved data"
    )
    parser.add_argument("--rank_id", type=int, default=0, 
                        help="Rank ID for distributed running.")
    parser.add_argument("--num_parallel", type=int, default=1, 
                        help="Number of parallel processes.")
    parser.add_argument("--num_devices", type=int, default=1, help="Number of available GPUs to run inference on.")
    return parser.parse_args()

def get_file_list(source):
    file_list = []
    if Path(source).suffix == ".txt":
        with open(source, "r") as f:
            file_list = [line.strip() for line in f.readlines()]  # Read file paths from the text file
    elif Path(source).suffix in [".jpg", ".png"]:
        file_list = [source]  # Single file case
    return file_list

def run_internvl(questions: list[str], modality: str, num_devices=1) -> ModelRequestData:
    assert modality == "image"
    model_name = "OpenGVLab/InternVL2_5-8B"
    attention_heads = 32
    print(largest_factor_less_than(num_devices, attention_heads))
    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=2048,
        limit_mm_per_prompt={modality: 1},
        tensor_parallel_size=largest_factor_less_than(num_devices, attention_heads)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [[{
        'role': 'user',
        'content': f"<image>\n{question}"
    }] for question in questions]
    prompts = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)
    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )

def guided_decoding(choices):
    guided_decoding_params = GuidedDecodingParams(choice=choices)
    choices_regex = "(" + "|".join(choices) + ")"
    return guided_decoding_params, choices_regex

class ImageLocationsPredictor():
    def __init__(self, num_devices=1):
        self.req_data = run_internvl([base_location_prompt,
                            base_activity_prompt], "image", num_devices)
        default_limits = {"image": 0, "video": 0, "audio": 0}
        self.req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
        self.req_data.engine_args.limit_mm_per_prompt or {})

        engine_args = asdict(self.req_data.engine_args) | {
            "seed": 10,
            "disable_mm_preprocessor_cache": False,
        }
        self.llm = LLM(**engine_args)
        self.location_prompt = self.req_data.prompts[0]
        self.activity_prompt = self.req_data.prompts[1]
        self.create_all_sampling_params()

    def create_sampling_params(self, guided_decoding_params, sampling_nums=sampling_nums, logprobs=logprobs):
        sampling_params = SamplingParams(
            n=sampling_nums,
            temperature=1.4,
            max_tokens=64,
            stop_token_ids=self.req_data.stop_token_ids,
            logprobs=logprobs
        )
        sampling_params.guided_decoding = guided_decoding_params
        return sampling_params

    def create_all_sampling_params(self):
        location_params, self.location_regex = guided_decoding(locations)
        activity_params, self.activity_regex = guided_decoding(activities)
        self.activity_sampling_params = self.create_sampling_params(activity_params)
        self.location_sampling_params = self.create_sampling_params(location_params)

    def predict_locations_activities(self, images, args):
        if os.path.exists(args.output):
            df = pd.read_csv(args.output)
            images = [image for image in images if image not in df["image_path"].values]
        total_prompt_batches = math.ceil(len(images) / args.prompting_batch)
        for batch in tqdm(range(total_prompt_batches), desc="Retrieving probabilities for prompts", total=total_prompt_batches):
            curr_images = batcher(images, batch+1, total_prompt_batches)
            location_generations = self.llm.generate(prompts=[{
                "prompt": self.location_prompt,
                "multi_modal_data": {"image": PIL.Image.open(curr_image).convert("RGB")}
                } for curr_image in curr_images], 
                sampling_params=self.location_sampling_params, use_tqdm=False)
            activity_generations = self.llm.generate(prompts=[{
                "prompt": self.activity_prompt,
                "multi_modal_data": {"image": PIL.Image.open(curr_image).convert("RGB")}
                } for curr_image in curr_images], 
                sampling_params=self.activity_sampling_params, use_tqdm=False)
            current_batch = assign_top_label_and_probs(location_generations, curr_images, "location", self.location_regex)
            current_batch = assign_top_label_and_probs(activity_generations, curr_images, "activity", self.activity_regex, current_batch=current_batch)  
            update_csv_with_batch_results(current_batch, curr_images, args.output)
        return len(images)

def main():
    args = parse_args()
    if not args.output:
        base = os.getcwd()
        args.output = Path(f"{base}/outputs")
    main_output_folder = args.output
    count = 0
    curr_goal = 1
    predictor = ImageLocationsPredictor(args.num_devices)
    if os.path.isdir(args.source):
        subdirs = [d for d in os.listdir(args.source) if os.path.isdir(os.path.join(args.source, d))]
        # If there are subdirectories assume that this means we want to save csv files at a video level/too much data to save a single CSV
        # Also ignoring parent directory level files, in the future could switch to using os.walk 
        if subdirs:
            number_of_subdirs = len(subdirs)
            group_size = number_of_subdirs // args.num_parallel
            start_idx = args.rank_id * group_size
            end_idx = start_idx + group_size
            if args.rank_id == args.num_parallel - 1:
                end_idx = number_of_subdirs
            current_group_frames = subdirs[start_idx:end_idx]
            for subdir in tqdm(current_group_frames, desc="Processing videos"):
                subdir_path = Path(args.source) / subdir
                args.output = Path(f'{main_output_folder}/{subdir}.csv')
                files_in_subdir = [str(file) for file in subdir_path.iterdir() 
                                if file.is_file() and (file.suffix in {'.jpg', '.png'})]
                count = count + predictor.predict_locations_activities(files_in_subdir, args)               
                if (count // 100000) == curr_goal:
                    print(f"Processed {count} images")
                    curr_goal = curr_goal + 1
        else:
            file_list = [str(Path(f"{args.source}/{file}")) for file in os.listdir(args.source)] 
            predictor.predict_locations_activities(file_list, args)
    else:
        predictor.predict_locations_activities(get_file_list(args.source), args)
        
if __name__ == "__main__":
    main()