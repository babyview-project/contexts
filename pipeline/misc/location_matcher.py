import pandas as pd
from sklearn.metrics import cohen_kappa_score, jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

# Load CSVs
df1_full = pd.read_csv('output.csv')
df2_full = pd.read_csv('output_new.csv')
#df2_full = pd.read_csv('/ccn2/dataset/babyview/outputs_20250312/locations/00320001_2024-05-04_1_f9a42af897_processed.csv')

# Define your key column (adjust as needed)
key_col = 'image_path'  # or 'id', etc.

# Store image_paths before setting index
common_keys = set(df1_full[key_col]).intersection(set(df2_full[key_col]))

# Filter to only common keys
df1 = df1_full[df1_full[key_col].isin(common_keys)].set_index(key_col)
df2 = df2_full[df2_full[key_col].isin(common_keys)].set_index(key_col)

# Basic checks
assert df1.shape[0] == df2.shape[0], "CSV files must have the same number of rows"
assert 'location' in df1.columns and 'location_options' in df1.columns, "Expected columns not found"
assert 'location' in df2.columns and 'location_options' in df2.columns, "Expected columns not found"

# Normalize location_options
def normalize_options(opt_str):
    if pd.isna(opt_str):
        return ''
    return ','.join(sorted(opt.strip() for opt in str(opt_str).split(',')))

df1['location_options_norm'] = df1['location_options'].apply(normalize_options)
df2['location_options_norm'] = df2['location_options'].apply(normalize_options)

# --- DIFFERENCE CHECKING ---
diff_mask = (df1['location'] != df2['location']) | (df1['location_options_norm'] != df2['location_options_norm'])

# Create comparison dataframe with differences and include image_path from index
comparison_df = pd.DataFrame({
    'image_path': diff_mask[diff_mask].index,  # Get image_paths from index where diff_mask is True
    'location_file1': df1.loc[diff_mask, 'location'],
    'location_file2': df2.loc[diff_mask, 'location'],
    'location_probs_file1': df1.loc[diff_mask, 'location_probs'] if 'location_probs' in df1.columns else 'N/A',
    'location_probs_file2': df2.loc[diff_mask, 'location_probs'] if 'location_probs' in df2.columns else 'N/A',
    'location_options_file1': df1.loc[diff_mask, 'location_options'],
    'location_options_file2': df2.loc[diff_mask, 'location_options'],
})
comparison_df.to_csv('location_differences_decoding.csv', index=False)

# --- CORRELATION METRICS ---

# 1. LOCATION exact match and Cohen's kappa
location_match_rate = (df1['location'] == df2['location']).mean()
location_kappa = cohen_kappa_score(df1['location'], df2['location'])

# 2. LOCATION_OPTIONS normalized match
location_opts_match_rate = (df1['location_options_norm'] == df2['location_options_norm']).mean()

# 3. LOCATION_OPTIONS Jaccard similarity
def jaccard_rowwise(a, b):
    set_a = set(str(a).split(',')) if pd.notna(a) and str(a).strip() else set()
    set_b = set(str(b).split(',')) if pd.notna(b) and str(b).strip() else set()
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)

jaccard_similarities = [
    jaccard_rowwise(a, b)
    for a, b in zip(df1['location_options'], df2['location_options'])
]
avg_jaccard = sum(jaccard_similarities) / len(jaccard_similarities)

# --- OUTPUT METRICS ---
print("=== LOCATION COLUMN ===")
print(f"Match rate: {location_match_rate:.2%}")
print(f"Cohen's kappa: {location_kappa:.3f}")

print("\n=== LOCATION_OPTIONS COLUMN ===")
print(f"Normalized match rate: {location_opts_match_rate:.2%}")
print(f"Average Jaccard similarity: {avg_jaccard:.3f}")