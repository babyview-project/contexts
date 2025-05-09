import re
from re import escape as regex_escape
from collections.abc import Sequence
import pandas as pd
import math
from vllm.model_executor.guided_decoding.outlines_logits_processors import RegexLogitsProcessor
import argparse
from tqdm import tqdm
from video_utils import batcher, softmax, update_csv_with_batch_results
import numpy as np
# from vllm.sampling_params import SamplingParams
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

activities = ["being held", "eating", "drinking", "playing with toy", "getting changed", "crawling", "crying", "exploring", "cleaning", "gardening", "watching tv", "driving", "reading", "nothing", "cooking"]
first_word_to_activity = {}
for activity in activities:
    first_word = activity.split()[0]
    first_word_to_activity[first_word] = activity

parser = argparse.ArgumentParser(description="Find probabilities of tokens using video descriptions and transcripts.")
parser.add_argument("--batch", type=int, default=1, help="Batch number of total batches being run, one-indexed.")
parser.add_argument("--total_batches", type=int, default=1, help="Total batches being run in parallel.")
parser.add_argument("--prompting_batch", type=int, default=1, help="How many prompts to pass into a single generate call")
args = parser.parse_args()
df = pd.read_csv("video_activities_locations_probs.csv")
current_batch = batcher(df, args.batch, args.total_batches) 
len(current_batch)
choices = [f"{regex_escape(str(choice))}" for choice in activities]
choices_regex = "(" + "|".join(choices) + ")"
guided_decoding_params = GuidedDecodingParams(choice=activities)  
end = f"options: {", ".join(activities)}? "
llm = LLM(model="OpenGVLab/InternVL2_5-8B", gpu_memory_utilization=0.9, enable_prefix_caching=True, trust_remote_code=True)
processors = [RegexLogitsProcessor(choices_regex, llm.get_tokenizer(), reasoner=None)]
sampling_nums = 5
logprobs = 20
set_seed = 10

vid_transcript_prompts = []
for idx, row in current_batch.iterrows():
    if "text_options" not in row or pd.isna(row["text_options"]) or row["text_options"] == "":
        vid_transcript_prompts.append((f"Video caption: {row["video_description"]}\nTranscript: {row["transcript"]}\nAnswer with one phrase what activity is going on in this video, taken with a camera attached to the head of a child, from the following options: {", ".join(activities)}? ", row["video_path"]))

total_prompt_batches = math.ceil(len(vid_transcript_prompts) / args.prompting_batch)
for batch in tqdm(range(total_prompt_batches), desc="Retrieving probabilities for prompts", total=total_prompt_batches):
    curr_prompts = batcher(vid_transcript_prompts, batch+1, total_prompt_batches)
    generations = llm.generate(prompts=[prompt[0] for prompt in curr_prompts], 
        sampling_params=SamplingParams(n=sampling_nums,logprobs=20,prompt_logprobs=None, 
                                guided_decoding=guided_decoding_params, stop="\n", seed=set_seed))
    vid_paths = [prompt[1] for prompt in curr_prompts]
    # here, the number of prompt_generations is based on how many prompts we're processing at once using prompting_batch    
    #  # here, the number of outputs is based on the sampling_nums column
    for prompt_generation, vid_path in zip(generations, vid_paths):
        # Store unique text options and their probabilities in a dictionary
        text_options_dict = {}
        # Store all samples and their probabilities in a dictionary
        samples_dict = {}
        for i, output in enumerate(prompt_generation.outputs):
            curr_text = output.text
            curr_prob = output.cumulative_logprob
            
            # Add to samples dictionary (no duplicates)
            text_options_dict[curr_text] = curr_prob
            
            if i == 0:
                # Add the first output text to text options
                text_options_dict[curr_text] = curr_prob
                
                # Add additional options from logprobs using regex pattern
                for logprob in output.logprobs[0].values():
                    if logprob.decoded_token in first_word_to_activity:
                        logprob.decoded_token = first_word_to_activity[logprob.decoded_token]
                    # Fix the regex matching using re.match
                    if re.match(choices_regex, logprob.decoded_token):
                        text_options_dict[logprob.decoded_token] = logprob.logprob
        
        # Convert to lists while maintaining the coupling
        text_options_list = list(text_options_dict.keys())
        probs_list = [text_options_dict[opt] for opt in text_options_list]
        
        #samples_list = list(samples_dict.keys())
        #sample_probs_list = [samples_dict[sample] for sample in samples_list]
        
        # Apply softmax to probabilities
        probs_softmax = softmax(np.array(probs_list))
        #sample_probs_softmax = softmax(np.array(sample_probs_list))
        
        # Create comma-separated strings
        text_options_str = ",".join(text_options_list)
        probs_str = ",".join([str(prob) for prob in probs_softmax])
        #samples_str = ",".join(samples_list)
        #sample_probs_str = ",".join([str(prob) for prob in sample_probs_softmax])
        
        # Update the dataframe with new columns
        mask = current_batch["video_path"] == vid_path
        current_batch.loc[mask, "text_options"] = text_options_str
        current_batch.loc[mask, "text_probs"] = probs_str
        #current_batch.loc[mask, "samples"] = samples_str
        #current_batch.loc[mask, "sample_probs"] = sample_probs_str
    
    update_csv_with_batch_results(current_batch, vid_paths)
