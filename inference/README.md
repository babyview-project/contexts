## Model choices
We are currently using [VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3) as our primary VQA model to retrieve locations, activities and video descriptions.
We have also used VideoChatFlash (InternVideo family) for initial video descriptions, InternVL2.5 for frame-level locations and for retrieving locations and activities using video descriptions.

## Chunking
Pipeline to chunk videos into one minute chunks
```
python -m chunking.chunker_parallel
```


