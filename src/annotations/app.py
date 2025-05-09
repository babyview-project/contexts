from flask import Flask, render_template, send_from_directory, request, Response, url_for, abort
from flask import abort, request

from dotenv import load_dotenv
import os
import pandas as pd
import re
import math

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
    "location": video_descriptions_raw["location"],
    "activity": video_descriptions_raw["activity_transcript"],
    "transcript": video_descriptions_raw["transcript"],
    "text": video_descriptions_raw["video_path"],
    "text1": [
        f"<b>Location: {loc}<br/>Activity (with transcript): {act_transcript}</b><br/>" +
        f"Video description + transcript: {lm}" +
        f"<br/>Text options: {text_options}<br/>Text probabilities: {text_probs}" +
        f"<br/><br/> Transcript: {transcript}"
        for loc, act, act_transcript, lm, transcript, text_options, text_probs in zip(
            video_descriptions_raw["location"],
            video_descriptions_raw["activity"],
            video_descriptions_raw["activity_transcript"],
            video_descriptions_raw["vid_transcript_lm"],
            video_descriptions_raw["transcript"],
            video_descriptions_raw["text_options"],
            video_descriptions_raw["text_probs"]
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
    # Search parameters
    search_term = request.args.get('search', '').lower()
    search_type = request.args.get('search_type', 'all')
    
    # Pagination parameters
    page = int(request.args.get('page', 1))
    per_page = 12  # Number of videos per page
    
    # Find all matches for the search first (across all pages)
    if search_term:
        all_matched_items = []
        
        for i, (path, loc, act, transcript, text) in enumerate(zip(
            video_descriptions["video_path"],
            video_descriptions["location"],
            video_descriptions["activity"],
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
            elif search_type == 'activity' and (safe_contains(act, search_term)):
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
    return send_from_directory(VIDEO_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)