import pandas as pd
import re
import os

df = pd.read_csv("results/EXP010/semantic_annotations_master.csv")

valid_tokens = [
    'pronoun', 'alias', 'discourse marker', 'paragraph break', 
    'scene break', 'explicit addressee', 'dialogue tag absent', 
    'topic continuation', 'narrative focus', 'other'
]

def fix_evidence(val):
    if pd.isna(val) or str(val).strip() == '': return val
    val = str(val).lower().replace(';', ',')
    # If the AI wrote "pronoun explicit addressee", we want "pronoun, explicit addressee"
    # We can just iterate through valid tokens and if they are in the string, collect them
    found = []
    for token in valid_tokens:
        if token in val:
            found.append(token)
    return ", ".join(found)

df['ANNOTATION: Evidence'] = df['ANNOTATION: Evidence'].apply(fix_evidence)
df.to_csv("results/EXP010/semantic_annotations_master.csv", index=False)

print("Evidence fixed.")
