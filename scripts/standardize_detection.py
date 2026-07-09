import pandas as pd
import os
import shutil

raw_path = "results/EXP010/semantic_annotations_master.csv"
backup_path = "results/EXP010/semantic_annotations_master_RAW.csv"

# Step 1: Freeze raw dataset
if not os.path.exists(backup_path):
    shutil.copy2(raw_path, backup_path)
    print(f"Backed up to {backup_path}")

df = pd.read_csv(raw_path)

def propose_mapping(col_name, unique_vals):
    print(f"### {col_name}")
    print("| Raw Value | Proposed Standard Value |")
    print("| :--- | :--- |")
    for val in unique_vals:
        if pd.isna(val): continue
        val_str = str(val).strip()
        mapped = val_str
        
        # Rule 3: Number prefix removal
        import re
        mapped = re.sub(r'^\d+[a-z]?\.\s*', '', mapped)
        
        # Capitalization rules for certain columns
        if "Confidence" in col_name:
            mapped = mapped.capitalize()
        if "Explicit Alternative" in col_name:
            if mapped.lower() in ['yes', 'no', 'partial', 'unknown', 'unsure']:
                mapped = mapped.capitalize()
                
        print(f"| `{val}` | `{mapped}` |")
    print()

cols = [
    'ANNOTATION: Primary Category',
    'ANNOTATION: Secondary Category',
    'ANNOTATION: Context Window Needed',
    'ANNOTATION: Confidence',
    'ANNOTATION: Explicit Alternative Feasible?'
]

print("## Proposed Mappings Report\n")
for col in cols:
    if col in df.columns:
        unique_vals = df[col].unique()
        propose_mapping(col, unique_vals)
