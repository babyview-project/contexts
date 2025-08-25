import re
from re import escape as regex_escape
from collections.abc import Sequence
from constants import get_activities, get_locations
import pandas as pd
import math
from vllm.model_executor.guided_decoding.outlines_logits_processors import RegexLogitsProcessor
import argparse
from tqdm import tqdm
from video_utils import batcher, softmax, update_csv_with_batch_results, assign_top_label_and_probs
import numpy as np
# from vllm.sampling_params import SamplingParams
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

activities = get_activities()
locations = get_locations()
# singing
first_word_to_activity = {}
for activity in activities:
    first_word = activity.split()[0]
    first_word_to_activity[first_word] = activity

parser = argparse.ArgumentParser(description="Find probabilities of tokens using video descriptions and transcripts.")
parser.add_argument("--batch", type=int, default=1, help="Batch number of total batches being run, one-indexed.")
parser.add_argument("--total_batches", type=int, default=1, help="Total batches being run in parallel.")
parser.add_argument("--prompting_batch", type=int, default=1, help="How many prompts to pass into a single generate call")
args = parser.parse_args()
df = pd.read_csv("vid_descriptions.csv")
current_batch = batcher(df, args.batch, args.total_batches) 
len(current_batch)
activity_choices = [f"{regex_escape(str(choice))}" for choice in activities]
activity_choices_regex = "(" + "|".join(activities) + ")"
location_choices_regex = "(" + "|".join(locations) + ")"
guided_decoding_params_activities = GuidedDecodingParams(choice=activities)  
guided_decoding_params_locations = GuidedDecodingParams(choice=locations)  
end = f"options: {", ".join(activities)}? "
llm = LLM(model="OpenGVLab/InternVL2_5-8B", gpu_memory_utilization=0.9, enable_prefix_caching=True, trust_remote_code=True)
#activity_processors = [RegexLogitsProcessor(activity_choices_regex, llm.get_tokenizer(), reasoner=None)]
sampling_nums = 10
logprobs = 20
set_seed = 10

vid_transcript_prompts = []
for idx, row in current_batch.iterrows():
    if "text_options" not in row or pd.isna(row["text_options"]) or row["text_options"] == "":
        vid_transcript_prompts.append([f"Video caption: {row["video_description"]}\nTranscript: {row["transcript"]}\nThis video is taken with a camera mounted to the head of a child. Answer with one phrase what activity the child with the camera on their head is participating in from the following options: {", ".join(activities)}? ",
                                       f"Video caption: {row["video_description"]}\nTranscript: {row["transcript"]}\nAnswer with one phrase what location this video is in, taken with a camera mounted to the head of a child, from the following options: {", ".join(locations)}? ", 
                                       row["video_path"]])

total_prompt_batches = math.ceil(len(vid_transcript_prompts) / args.prompting_batch)
for batch in tqdm(range(total_prompt_batches), desc="Retrieving probabilities for prompts", total=total_prompt_batches):
    curr_prompts = batcher(vid_transcript_prompts, batch+1, total_prompt_batches)
    # 0 = location prompt, 1 = activity prompt, 2 = video path
    activity_generations = llm.generate(prompts=[prompt[0] for prompt in curr_prompts], 
        sampling_params=SamplingParams(n=sampling_nums,logprobs=20,prompt_logprobs=None, 
                                guided_decoding=guided_decoding_params_activities, stop="\n", seed=set_seed), use_tqdm=False)
    location_generations = llm.generate(prompts=[prompt[1] for prompt in curr_prompts], 
        sampling_params=SamplingParams(n=sampling_nums,logprobs=20,prompt_logprobs=None, 
                                guided_decoding=guided_decoding_params_locations, stop="\n", seed=set_seed), use_tqdm=False)
    vid_paths = [prompt[2] for prompt in curr_prompts]
    current_batch = assign_top_label_and_probs(location_generations, vid_paths, "location", location_choices_regex)
    current_batch = assign_top_label_and_probs(activity_generations, vid_paths, "activity", activity_choices_regex, current_batch=current_batch) 
    current_batch = current_batch.rename(columns={'image_path': 'video_path'})
    update_csv_with_batch_results(current_batch, vid_paths, "video_activities_locations_probs_0604.csv", keypath="video_path")
