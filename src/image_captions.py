import csv
from tqdm import tqdm
from models.image.llava_onevision import LLAVAOneVision

def batch_list(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
image_captioner = LLAVAOneVision()
# Step 1: Read image paths
with open('/ccn2/dataset/babyview/outputs_20250312/10000_random_frames.txt', 'r') as file:
    paths = [line.strip() for line in file if line.strip()]

# Step 2: Generate captions in batches
tenkframe_descriptions = []
batches = batch_list(paths, 1)

for batch in tqdm(batches, desc="Image descriptions"):
    captions = image_captioner.caption_images(batch)
    tenkframe_descriptions.extend(captions)

# Step 3: Save to CSV
csv_path = 'tenkframe_descriptions.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'caption'])  # header
    for path, caption in zip(paths, tenkframe_descriptions):
        writer.writerow([path, caption])

print(f"Saved {len(tenkframe_descriptions)} captions to {csv_path}")
