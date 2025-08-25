import pandas as pd
def get_locations():
    df = pd.read_csv("location_schema.csv")
    return df.loc[df["proposal"] == 1, "location"].tolist()

def get_activities():
    df = pd.read_csv("activity_schema.csv")
    return df.loc[df["proposal"] == 1, "activity"].tolist()

def get_original_locations():
    return ["bathroom", "bedroom", "car", "closet", "garage", "living room", "hallway", "outside", "garage", "kitchen", "deck"]

def get_original_activities():
    return ["being held", "eating", "drinking", "playing with toy", "getting changed", "crawling", "crying"]

