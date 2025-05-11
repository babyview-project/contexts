# Scene descriptions
This repository stores code to describe scenes, locations, and activities present in the BabyView dataset.

## Running any Python code
```
conda create --name activitycap python=3.12
conda activate activitycap
pip install -r requirements.txt
```

## Frame predictions
```
cd src
python image_locations_probs_parallel.py --device_ids [0,1,2,3,4,5,6,7] --num_parallel 2
```

## Gallery running
Set username and password in your `.env` file 

Run locally:
```
cd src/annotations
python app.py
```

Run static website
```
cd src/annotations
gunicorn -w 4 app:app --bind 0.0.0.0:<PORT>
```