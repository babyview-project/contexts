import os
token = os.environ['HF_TOKEN']
import torch

from transformers import AutoTokenizer, AutoModel


tokenizer =  AutoTokenizer.from_pretrained('OpenGVLab/InternVideo2-Chat-8B', trust_remote_code=True, use_fast=False)

model = AutoModel.from_pretrained(
    'OpenGVLab/InternVideo2-Chat-8B',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True).cuda()

from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import numpy as np
import decord
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
decord.bridge.set_bridge("torch")

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)
    frames = transform(frames)

    T_, C, H, W = frames.shape
        
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames

video_path = "/labs/vislearnlab/experiments/activitycap/data/babyview-example-3.mp4"
# sample uniformly 96 frames from the video
video_tensor = load_video(video_path, num_segments=48, return_msg=False)
video_tensor = video_tensor.to(model.device)

chat_history= []
response, chat_history = model.chat(tokenizer, '', 'summarize the video.', media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True,generation_config={'do_sample':False})
print(response)
# The video shows a woman performing yoga on a rooftop with a beautiful view of the mountains in the background. She starts by standing on her hands and knees, then moves into a downward dog position, and finally ends with a standing position. Throughout the video, she maintains a steady and fluid movement, focusing on her breath and alignment. The video is a great example of how yoga can be practiced in different environments and how it can be a great way to connect with nature and find inner peace.

response, chat_history = model.chat(tokenizer, '', 'What instrument is being used?', media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True,generation_config={'do_sample':False})
# The woman in the video is wearing a black tank top and grey yoga pants.
print(response)
