from flask import Flask, render_template, send_from_directory, request, Response, url_for, abort
from flask import abort, request

from dotenv import load_dotenv
import os
import pandas as pd
import re
from pathlib import Path
import math

# Load credentials from .env
load_dotenv()
USERNAME = os.getenv("APP_USERNAME")
PASSWORD = os.getenv("APP_PASSWORD")
PROJECT_PATH = os.getenv("PROJECT_PATH")
app = Flask(__name__)

VIDEO_DIR = "/"
FULL_VIDEO_DIR = "/ccn2/dataset/babyview/unzip_2025/babyview_main_storage"
IMAGE_DIR = "/"

# Load data
image_descriptions_raw = pd.read_csv(f"{PROJECT_PATH}/data/image_activities_locations.csv")
video_descriptions_raw = pd.read_csv(f"{PROJECT_PATH}/data/full_video_activities_locations.csv") #"/home/tsepuri/activitycap/pipeline/data/video_activities_locations_all.csv"

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
    "location_llm": video_descriptions_raw["location_options"],
    #"activity": video_descriptions_raw["activity_transcript"],
    "transcript": video_descriptions_raw["transcript"],
    "activity_llm": video_descriptions_raw["activity_options"],
    "text1": video_descriptions_raw["video_path"],
    "text": [
        f"<b>Locations from LLM: {loc_options}<br/>LLM probabilities: {loc_probs}" +
        f"<br/>Activities from LLM: {act_options}<br/>LLM probabilities: {act_probs}" +
        f"<br/>Location from VQA: {loc_vqa}<br/>Activity from VQA: {act_vqa}</b><br/>" +
        f"<br/><br/> Transcript: {transcript}"
        for loc_vqa, act_vqa, transcript, loc_options, loc_probs, act_options, act_probs in zip(
            video_descriptions_raw["location_vqa"],
            video_descriptions_raw["activity_vqa"],
            video_descriptions_raw["transcript"],
            video_descriptions_raw["location_options"],
            video_descriptions_raw["location_probs"],
            video_descriptions_raw["activity_options"],
            video_descriptions_raw["activity_options"]
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

@app.route("/video_loader")
def video_loader():
    search_term = request.args.get("search", "")
    video_path = search_term
    video_found = False
    if search_term:
        video_path = Path(f"{FULL_VIDEO_DIR}/{search_term}/{search_term}_processed.MP4")
        if os.path.exists(video_path):
            video_found = True

    return render_template("video_loader.html",
                           search_term=search_term,
                           video_path=video_path,
                           video_found=video_found)

@app.route("/videos")
@requires_auth
def videos():
    # Search parameters
    search_term = request.args.get('search', '').lower()
    search_type = request.args.get('search_type', 'all')
    
    # Pagination parameters
    page = int(request.args.get('page', 1))
    per_page = 12  # Number of videos per page
    
    # Find all matches for the search first (across all pages)
    if search_term:
        all_matched_items = []
        
        for i, (path, loc, act_llm, transcript, text) in enumerate(zip(
            video_descriptions["video_path"],
            video_descriptions["location_llm"],
            #video_descriptions["activity"],
            video_descriptions["activity_llm"],
            video_descriptions["transcript"],
            video_descriptions["text"]
        )):
            # Apply different search criteria based on search type
            match = False
            
            # Helper function to safely convert to string and check for substring
            def safe_contains(value, search_substring):
                if value is None:
                    return False
                # Convert to string if not already a string
                if not isinstance(value, str):
                    value = str(value)
                return search_substring in value.lower()
            
            if search_type == 'transcript' and safe_contains(transcript, search_term):
                match = True
            elif search_type == 'location' and (safe_contains(loc, search_term)):
                match = True
            elif search_type == "activity_llm" and (safe_contains(act_llm, search_term)):
                match = True
            elif search_type == "video_path" and (safe_contains(path, search_term)):
                match = True
            elif search_type == 'all' and (safe_contains(text, search_term)):
                match = True
                
            if match:
                # Ensure text is a string before highlighting
                text_str = text if isinstance(text, str) else str(text)
                
                # Highlight the search term in the text
                highlighted_text = re.sub(
                    f'({re.escape(search_term)})', 
                    r'<span style="background-color: yellow; font-weight: bold;">\1</span>', 
                    text_str, 
                    flags=re.IGNORECASE
                )
                all_matched_items.append((path, highlighted_text))
    else:
        all_matched_items = list(zip(video_descriptions["video_path"], video_descriptions["text"]))
    
    # Calculate total pages
    total_items = len(all_matched_items)
    total_pages = math.ceil(total_items / per_page)
    
    # Ensure page is within valid range
    if page < 1:
        page = 1
    elif page > total_pages and total_pages > 0:
        page = total_pages
        
    # Get items for current page
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    page_items = all_matched_items[start_index:end_index]
    
    # Generate pagination links for template
    def generate_page_url(page_num):
        args = request.args.copy()
        args['page'] = page_num
        return url_for('videos', **args)
    
    return render_template("video_gallery.html", 
                          items=page_items,
                          search_term=search_term,
                          search_type=search_type,
                          page=page,
                          total_pages=total_pages,
                          total_items=total_items,
                          prev_url=generate_page_url(page-1) if page > 1 else None,
                          next_url=generate_page_url(page+1) if page < total_pages else None,
                          page_urls=[generate_page_url(i) for i in range(max(1, page-2), min(total_pages+1, page+3))])

@app.route("/images/<path:filename>")
@requires_auth
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route("/videos/<path:filename>")
@requires_auth
def serve_video(filename):
    response = send_from_directory(VIDEO_DIR, filename)
    response.headers.add('Cache-Control', 'public, max-age=3600')
    response.headers.add('Accept-Ranges', 'bytes')
    return response

if __name__ == "__main__":
    app.run(debug=True)