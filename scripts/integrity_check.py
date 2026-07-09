import pandas as pd
import os

df = pd.read_csv("results/EXP010/semantic_annotations_master.csv")
df_raw = pd.read_csv("results/EXP010/semantic_annotations_master_RAW.csv")

# Fix massive note on PrideAndPrejudice_61
for idx, row in df.iterrows():
    if row['Quote_ID'] == 'PrideAndPrejudice_61':
        note = str(row['ANNOTATION: Notes'])
        if "--- Batch 008 Summary" in note:
            # truncate it
            clean_note = note.split("--- Batch 008 Summary")[0].strip()
            df.at[idx, 'ANNOTATION: Notes'] = clean_note

df.to_csv("results/EXP010/semantic_annotations_master.csv", index=False)

print("Integrity Check Report:")
print("1. Total Rows:")
print(f"   RAW: {len(df_raw)} rows")
print(f"   NORMALIZED: {len(df)} rows")
print(f"   Pass? {'Yes' if len(df) == 200 and len(df) == len(df_raw) else 'No'}")

print("\n2. Missing values in required columns (Primary Category):")
missing = df['ANNOTATION: Primary Category'].isna().sum()
print(f"   Missing Primary Categories: {missing}")
print(f"   Pass? {'Yes' if missing == 0 else 'No'}")

print("\n3. Valid Primary Categories:")
cats = set(df['ANNOTATION: Primary Category'].dropna().unique())
valid_cats = {
    'Reference: Pronominal Coreference',
    'Reference: Lexical Normalization / Alias Matching',
    'Discourse: Speaker Continuity',
    'Pragmatics: Speaker–Addressee Semantics',
    'Discourse Structure: Scene Transitions',
    'Other / Unclassified'
}
invalid_cats = cats - valid_cats
print(f"   Found Categories: {cats}")
print(f"   Invalid Categories: {invalid_cats if invalid_cats else 'None'}")
print(f"   Pass? {'Yes' if not invalid_cats else 'No'}")

print("\n4. Valid Evidence Labels:")
valid_evidence_tokens = {
    'pronoun', 'alias', 'discourse marker', 'paragraph break', 
    'scene break', 'explicit addressee', 'dialogue tag absent', 
    'topic continuation', 'narrative focus', 'other'
}

all_evidences = df['ANNOTATION: Evidence'].dropna().unique()
invalid_tokens = set()
for ev_str in all_evidences:
    tokens = [t.strip().lower() for t in ev_str.replace(';', ',').split(',')]
    for t in tokens:
        if t not in valid_evidence_tokens and t != '':
            invalid_tokens.add(t)

print(f"   Invalid Evidence Tokens: {invalid_tokens if invalid_tokens else 'None'}")
print(f"   Pass? {'Yes' if not invalid_tokens else 'No'}")

print("\n5. RAW Backup Exists:")
exists = os.path.exists("results/EXP010/semantic_annotations_master_RAW.csv")
print(f"   Exists? {'Yes' if exists else 'No'}")
