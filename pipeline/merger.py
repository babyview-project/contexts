import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

input_dir = Path("/ccn2/dataset/babyview/outputs_20250312/yoloe/unconstrained")
output_file = input_dir / "merged_output.csv"
csv_files = list(input_dir.glob("*.csv"))

def read_csv(path):
    try:
        # Try pyarrow engine for faster parsing
        return pd.read_csv(path, engine="pyarrow")
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            return None

# Parallel read and write to a single file in streaming fashion
with open(output_file, 'w') as f_out:
    header_written = False
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(read_csv, file): file for file in csv_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Merging CSVs"):
            df = future.result()
            if df is not None:
                df.to_csv(f_out, index=False, header=not header_written)
                header_written = True
