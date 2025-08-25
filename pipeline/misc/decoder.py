import re
from re import escape as regex_escape
from collections.abc import Sequence
import pandas as pd
from vllm.model_executor.guided_decoding.outlines_logits_processors import RegexLogitsProcessor
# from vllm.sampling_params import SamplingParams
from vllm import LLM, SamplingParams
from decoding.generators import BestOfN
from decoding.models import LanguageModel
from decoding.scorers import Scorer

activities = ["being held", "eating", "drinking", "playing with toy", "getting changed", "crawling", "crying", "exploring", "cleaning", "gardening", "watching tv", "driving", "reading", "nothing", "cooking"]

# load your small and big models
llm_small = LanguageModel.from_id(
    "allenai/OLMo-1B-hf", gpu_memory_utilization=0.2, enable_prefix_caching=True
)

llm_large = LanguageModel.from_id(
    "OpenGVLab/InternVL2_5-8B", gpu_memory_utilization=0.9, enable_prefix_caching=True, trust_remote_code=True
)

#llm_test = LLM(id="OpenGVLab/InternVL2_5-8B", gpu_memory_utilization=0.6, )
# since we're using vLLM under the hood,
# we can specify GPU memory utilization,
# and take advantage of prefix KV caching,
# among other optimizations

# write a scoring function and construct a scoring object
def score_fn(prompts: Sequence[str]) -> list[float]:
    print(prompts)
    contexts = [f"{prompt.split(end)[0]}{end}" for prompt in prompts]
    queries = [prompt.split(end)[1] for prompt in prompts]
    print(queries)
    logps = -llm_large.surprise(contexts=contexts, queries=queries)
    return logps.tolist()

scorer = Scorer.from_f_batch_str_to_batch_num(score_fn)

df = pd.read_csv("/home/tsepuri/activitycap/src/video_activities_locations.csv")
vid_transcript_lm_prompt = f"Video caption: {df["video_description"][10]}\nTranscript: {df["transcript"][10]}\nAnswer with one phrase what activity is going on in this video, taken with a camera attached to the head of a child, from the following options: {", ".join(activities)}? "
'''
vid_transcript_lm_prompt = f"""Please answer the question below about the video given its scene description and audio transcript.
Scene description: {df["video_description"][0]}
Audio transcript: {df["transcript"][0]}
What activity is happening in this video, taken with a camera attached to the head of a child, from the following options: {", ".join(activities)}? """
'''
end = f"options: {", ".join(activities)}? "
choices = [regex_escape(str(choice)) for choice in activities]
choices_regex = "(" + "|".join(choices) + ")"

#pattern = r" \d+ [\+\-] \d+\n"
processors = [RegexLogitsProcessor(choices_regex, llm_large.tokenizer)]

# and let's wrap this up with a `BestOfN` generator that will:
# - sample n generations from the small model
# - return them re-reranked by the big model
# we'll just take the top output, and return its value

def bon(prompt: str, n: int) -> str:
    return BestOfN(
        prompt=prompt,
        llm=llm_small,
        scorer=scorer,
        n=n,
        stop_str="\n",
        logits_processors=processors,
        seed=34,
    )

a = llm_large(prompts=[vid_transcript_lm_prompt], 
              params=SamplingParams(n=3,best_of=3,logprobs=10,prompt_logprobs=0, 
                                    logits_processors=processors, stop="\n", seed=5,
                                    include_stop_str_in_output=True))
                                    
print(a)
# let's run with n=5 and see what we get
out = bon(vid_transcript_lm_prompt, n=3)
for scoreditem in out:
    item = scoreditem.item.split(end)[1]
    score = scoreditem.score
    print(f"Item: {item} Score: {score}")
#expr = re.findall(pattern, out)[0].strip()


# looks good