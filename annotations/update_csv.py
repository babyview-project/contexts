import pandas as pd
from pathlib import Path

# Read the CSV file
df = pd.read_csv('data/clip_alignment.csv')

# Rename the original image_path column to original_image_path
#df['original_image_path'] = df['image_path']

# Create new image_path column with just the filename
def create_new_path(img_path):
    if pd.isna(img_path):
        return img_path
    
    source = Path(img_path)
    parent_dir = source.parent.name
    original_name = source.stem
    extension = source.suffix
    new_filename = f"{parent_dir}_{original_name}{extension}"
    
    return new_filename

#df['image_path'] = df['original_image_path'].apply(create_new_path)

for col in ['distractor_img1', 'distractor_img2', 'distractor_img3']:
    if col in df.columns:
        # Save original paths
        df[f'original_{col}'] = df[col]
        # Update with new paths
        df[col] = df[f'original_{col}'].apply(create_new_path)

# Save the updated CSV
df.to_csv('data/clip_alignment.csv', index=False)

print("Done!")
print(f"Updated {len(df)} rows")
print(f"\nFirst few entries:")
print(df[['image_path', 'original_image_path']].head())