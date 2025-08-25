import pandas as pd

# Load recordings CSV
recordings = pd.read_csv('/home/tsepuri/activitycap/data/recordings_processed.csv')

# Basic info
print(f"Total rows in recordings: {len(recordings)}")

# Check for missing values
missing_vals = recordings["superseded_gcp_name_feb25"].isna().sum()
print(f"Missing (NaN) values: {missing_vals}")

# Check number of raw unique values (may include whitespace, etc.)
raw_unique = recordings["superseded_gcp_name_feb25"].nunique()
print(f"Raw unique IDs: {raw_unique}")

# Check number of unique values after cleaning whitespace
cleaned_ids = recordings["superseded_gcp_name_feb25"].astype(str).str.strip()
cleaned_unique = cleaned_ids.nunique()
print(f"Unique IDs after strip(): {cleaned_unique}")

# Show a few problematic examples, if any
if cleaned_unique != raw_unique:
    print("Examples of potential duplicates after cleaning:")
    dupes = recordings["superseded_gcp_name_feb25"].value_counts()
    print(dupes[dupes > 1].head(10))

# Final home_video_ids set
home_video_ids = set(cleaned_ids)
print(f"Final set length: {len(home_video_ids)}")
