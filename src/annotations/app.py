from flask import Flask, render_template, send_from_directory, request, Response
from dotenv import load_dotenv
import os
import pandas as pd

# Load credentials from .env
load_dotenv()
USERNAME = os.getenv("APP_USERNAME")
PASSWORD = os.getenv("APP_PASSWORD")

app = Flask(__name__)

VIDEO_DIR = "/"
IMAGE_DIR = "/"

# Load data
image_descriptions_raw = pd.read_csv("/home/tsepuri/activitycap/src/image_activities_locations.csv")
video_descriptions_raw = pd.read_csv("/home/tsepuri/activitycap/src/video_activities_locations_probs.csv")

image_descriptions = {
    "image_path": image_descriptions_raw["image_path"],
    "text": [
        f"<b>Location: {loc}<br/>Activity: {act}</b><br/><br/>Caption: {cap}"
        for loc, act, cap in zip(
            image_descriptions_raw["location"],
            image_descriptions_raw["activity"],
            image_descriptions_raw["caption"]
        )
    ]
}

video_descriptions = {
    "video_path": video_descriptions_raw["video_path"],
    "text1": video_descriptions_raw["video_path"],
    "text": [
        f"<b>Location: {loc}<br/>Activity (with transcript): {act_transcript}</b><br/>" +
        f"Activity (with no transcript): {act}<br/>Video description + transcript: {lm}" +
        f"<br/>Samples: {samples}<br/>Sample probabilities: {sample_probs}" +
        f"<br/>Log prob texts: {text_options}<br/>Text probabilities: {text_probs}" +
        f"<br/><br/> Transcript: {transcript}"
        for loc, act, act_transcript, lm, transcript, text_options, text_probs, samples, sample_probs in zip(
            video_descriptions_raw["location"],
            video_descriptions_raw["activity"],
            video_descriptions_raw["activity_transcript"],
            video_descriptions_raw["vid_transcript_lm"],
            video_descriptions_raw["transcript"],
            video_descriptions_raw["text_options"],
            video_descriptions_raw["text_probs"],
            video_descriptions_raw["samples"],
            video_descriptions_raw["sample_probs"]
        )
    ]
}

# Authentication check
def check_auth(username, password):
    return username == USERNAME and password == PASSWORD

def authenticate():
    return Response(
        'Could not verify your access. Please log in.\n',
        401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    decorated.__name__ = f.__name__
    return decorated

@app.route("/")
@requires_auth
def index():
    items = zip(image_descriptions["image_path"], image_descriptions["text"])
    return render_template("image_gallery.html", items=items)

@app.route("/videos")
@requires_auth
def videos():
    items = zip(video_descriptions["video_path"], video_descriptions["text"])
    return render_template("video_gallery.html", items=items)

@app.route("/images/<path:filename>")
@requires_auth
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route("/videos/<path:filename>")
@requires_auth
def serve_video(filename):
    return send_from_directory(VIDEO_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)
