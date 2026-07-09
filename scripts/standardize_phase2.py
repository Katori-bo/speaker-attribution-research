import pandas as pd
import re
import numpy as np
import shutil

df = pd.read_csv("results/EXP010/semantic_annotations_master.csv")

# 1. Handle Legacy Category
def map_legacy_category(row):
    cat = str(row['ANNOTATION: Primary Category']).strip()
    if pd.isna(row['ANNOTATION: Primary Category']) or cat == 'nan':
        return row['ANNOTATION: Primary Category']
        
    if "Reference: Coreference and Aliasing" in cat:
        evidence = str(row['ANNOTATION: Evidence']).lower()
        if 'alias' in evidence:
            return "Reference: Lexical Normalization / Alias Matching"
        elif 'pronoun' in evidence or 'discourse marker' in evidence:
            return "Reference: Pronominal Coreference"
        else:
            return "Reference: Pronominal Coreference" # Default fallback for legacy reference
    return cat

df['ANNOTATION: Primary Category'] = df.apply(map_legacy_category, axis=1)

# 2. Normalize categories (remove numbering, case, whitespace)
def normalize_text(text):
    if pd.isna(text) or str(text) == 'nan':
        return np.nan
    text = str(text).strip()
    # Remove numbering prefix like "1a. ", "2. "
    text = re.sub(r'^\d+[a-zA-Z]?\.\s*', '', text)
    return text

df['ANNOTATION: Primary Category'] = df['ANNOTATION: Primary Category'].apply(normalize_text)
df['ANNOTATION: Secondary Category'] = df['ANNOTATION: Secondary Category'].apply(normalize_text)

# 3. Normalize Confidence
def normalize_confidence(text):
    if pd.isna(text) or str(text) == 'nan':
        return np.nan
    return str(text).strip().capitalize()
df['ANNOTATION: Confidence'] = df['ANNOTATION: Confidence'].apply(normalize_confidence)

# 4. Normalize Context Window Needed
df['ANNOTATION: Context Window Needed'] = df['ANNOTATION: Context Window Needed'].apply(
    lambda x: str(x).strip() if pd.notna(x) and str(x) != 'nan' else np.nan
)

# 5. Handle Hallucinated text and Normalize 'Explicit Alternative Feasible?'
def normalize_feasible(row):
    val = row['ANNOTATION: Explicit Alternative Feasible?']
    if pd.isna(val) or str(val) == 'nan':
        return val, row['ANNOTATION: Notes']
        
    val_str = str(val).strip()
    # Handle the specific hallucinated paragraph
    if val_str.startswith("No For Quotes 1, 3, 4"):
        new_val = "Yes"
        new_note = str(row['ANNOTATION: Notes']) + " | Original Feasible Column: " + val_str if pd.notna(row['ANNOTATION: Notes']) else "Original Feasible Column: " + val_str
        return new_val, new_note
        
    # Standardize casing for valid values
    if val_str.lower() in ['yes', 'no', 'partial', 'unknown', 'unsure']:
        return val_str.capitalize(), row['ANNOTATION: Notes']
        
    return val_str, row['ANNOTATION: Notes']

results = df.apply(normalize_feasible, axis=1)
df['ANNOTATION: Explicit Alternative Feasible?'] = [r[0] for r in results]
df['ANNOTATION: Notes'] = [r[1] for r in results]

# 6. Validate
print("Primary Categories:", df['ANNOTATION: Primary Category'].dropna().unique())
print("Confidence:", df['ANNOTATION: Confidence'].dropna().unique())
print("Feasible:", df['ANNOTATION: Explicit Alternative Feasible?'].dropna().unique())

df.to_csv("results/EXP010/semantic_annotations_master_NORMALIZED.csv", index=False)
shutil.copy2("results/EXP010/semantic_annotations_master_NORMALIZED.csv", "results/EXP010/semantic_annotations_master.csv")
print("Normalization complete.")
