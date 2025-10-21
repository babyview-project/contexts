# Contexts
This repository stores data processing and analysis code used to characterize the activities and locations present in the BabyView dataset.

## Data
The extracted activities and locations can be found in the `all_contexts.csv.zip` file within the `data` folder. We are filtering the video descriptions to ensure we only include descriptions that do not include any identifiable data.

## Analysis
All analysis code is in the `analysis` folder, `main_analysis.qmd` is the primary file of interest.

## Inference
All code used to run and compare various models and extract activities and locations from the BabyView dataset can be found in the `inference` folder.

### Running any Python code
```
conda create --name contexts python=3.12
conda activate contexts
pip install -r requirements.txt
```

### Video level predictions
```
python constrained_contexts.py --output_path /ccn2/dataset/babyview/outputs_20250312/activities
```

### Frame predictions
```
cd inference
python frame_level.image_locations_probs_parallel.py --device_ids [0,1,2,3,4,5,6,7] --num_parallel 2
```

## Gallery running
A gallery for examining detected activities and locations alongside utterances, clips, or frames in a web browser can be found in the `gallery` folder.

Set username and password in your `.env` file 

Run locally:
```
cd gallery
python app.py
```

Run static website
```
cd gallery
gunicorn -w 4 app:app --bind 0.0.0.0:<PORT>
```
