import pandas as pd
df1 = pd.read_csv("video_activities_locations_probs.csv")
df2 = pd.read_csv("video_activities_locations_probs_0508.csv")
df1 = df1.drop_duplicates(subset='video_path', keep='first')
df2 = df2.drop_duplicates(subset='video_path', keep='first')

# Concatenate, preferring df2 by putting it first
merged = pd.concat([df2, df1])

# Drop duplicates again to ensure uniqueness, keeping df2's version
merged = merged.drop_duplicates(subset='video_path', keep='first')

# Print counts
print("Unique video paths in df1:", df1['video_path'].nunique())
print("Unique video paths in df2:", df2['video_path'].nunique())
print("Unique video paths in merged df:", merged['video_path'].nunique())
merged.to_csv("video_activities_locations_all.csv")