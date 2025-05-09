from tqdm import tqdm
import pandas as pd
from models.image.internvl import InternVL
locations = ["bathroom", "bedroom", "car", "closet", "garage", "living room", "hallway", "outside", "garage", "kitchen", "deck"]
activities = ["being held", "eating", "drinking", "playing with toy", "getting changed", "crawling", "crying", "exploring"]
image_captioner_2 = InternVL()
descriptions = pd.read_csv("tenkframe_descriptions.csv")
df = []
for i, (image_path, caption) in tqdm(enumerate(zip(descriptions["image_path"], descriptions["caption"])), desc="Getting image descriptions"):
    location = image_captioner_2.caption_image(image_path,f"Answer with one word what location this image is in from the following: {", ".join(locations)}")
    activity = image_captioner_2.caption_image(image_path,f"Answer with one word what activity is going on in this image in from the following: {", ".join(activities)}")
    df.append({
        "image_path": image_path,
        "location": location,
        "activity": activity,
        "caption": caption
    })
    pd.DataFrame(df).to_csv("image_activities_locations_10k.csv", index=False)