import pandas as pd
from pathlib import Path
from src.coreference.pipeline import SemanticFeatureProvider

df = pd.read_csv("data/raw/pdnc/phase2/candidate_features.csv")
p = SemanticFeatureProvider()
df = df.head(1000).copy()
out = p.augment_features(df)
print("Unmapped:", (out['nearest_coref_dist'] == -1).sum(), "out of", len(out))
