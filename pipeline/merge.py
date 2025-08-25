import pandas as pd
df1 = pd.read_csv("vid_activities_0.csv")
df2 = pd.read_csv("data/video_activities_locations_probs_0604.csv")
print(len(df2))
print(df1[df1['video_path'].duplicated()].head())

merged_df = pd.merge(df1, df2, on='video_path', how='right')
print(len(merged_df))
merged_df.to_csv("data/full_video_activities_locations.csv")
