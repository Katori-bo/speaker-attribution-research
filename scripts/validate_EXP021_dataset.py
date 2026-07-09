import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import random
import torch
import numpy as np
import os

from src.evaluation.runner import load_frozen_exp014_dataset
from src.neural.dataset import SpeakerSequenceDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading frozen EXP014 dataset...")
    df = load_frozen_exp014_dataset()
    
    # We only care about the features used in EXP014
    base_feats = [c for c in df.columns if c not in [
        "quote_id", "novel", "candidate", "gold_speaker", "split", "label",
        "quote_start_byte", "quote_end_byte", "quoteByteSpans",
        "candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency",
        "candidate_is_attributed_speaker"
    ] and not c.startswith("symbolic_")]

    feature_cols = base_feats + [
        "candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency",
        "candidate_is_attributed_speaker"
    ]
    feature_cols = sorted(feature_cols)
    
    logger.info("Building SpeakerSequenceDataset...")
    dataset = SpeakerSequenceDataset(df, feature_cols)
    
    logger.info("Running validations...")
    
    # 1. Number of quotes before conversion == number after conversion
    original_quote_count = df['quote_id'].nunique()
    new_quote_count = sum(len(seq.quotes) for seq in dataset)
    
    # 2. Candidate count unchanged
    original_candidate_count = len(df)
    new_candidate_count = sum(len(q.candidates) for seq in dataset for q in seq.quotes)
    
    # 3. Gold speaker availability unchanged
    original_gold_count = df['label'].sum()
    new_gold_count = sum(1 for seq in dataset for q in seq.quotes for c in q.candidates if c.is_gold)
    
    # 4. Feature vector dimension unchanged
    expected_dim = len(feature_cols)
    actual_dim = dataset[0].quotes[0].candidates[0].features.shape[0]
    
    # 5. Randomly sample 100 candidates: Original feature vector == PyTorch tensor values
    all_candidates_flat = []
    for seq in dataset:
        for q in seq.quotes:
            for c in q.candidates:
                all_candidates_flat.append({
                    "quote_id": q.quote_id,
                    "candidate": c.candidate_id,
                    "features": c.features
                })
    
    random.seed(42)
    sample_indices = random.sample(range(len(all_candidates_flat)), 100)
    
    tensor_match = True
    for idx in sample_indices:
        item = all_candidates_flat[idx]
        q_id = item['quote_id']
        cand_id = item['candidate']
        t_feats = item['features'].numpy()
        
        orig_row = df[(df['quote_id'] == q_id) & (df['candidate'] == cand_id)].iloc[0]
        orig_feats = np.array([orig_row[c] for c in feature_cols], dtype=np.float32)
        
        # Replace NaNs with 0 for comparison if they exist, though EXP014 features shouldn't have NaNs
        t_feats = np.nan_to_num(t_feats)
        orig_feats = np.nan_to_num(orig_feats)
        
        if not np.allclose(t_feats, orig_feats, atol=1e-5):
            tensor_match = False
            break
            
    results = {
        "quotes_match": bool(original_quote_count == new_quote_count),
        "original_quotes": int(original_quote_count),
        "new_quotes": int(new_quote_count),
        
        "candidates_match": bool(original_candidate_count == new_candidate_count),
        "original_candidates": int(original_candidate_count),
        "new_candidates": int(new_candidate_count),
        
        "gold_match": bool(original_gold_count == new_gold_count),
        "original_gold_count": int(original_gold_count),
        "new_gold_count": int(new_gold_count),
        
        "dimension_match": bool(expected_dim == actual_dim),
        "expected_dim": int(expected_dim),
        "actual_dim": int(actual_dim),
        
        "tensor_match": bool(tensor_match)
    }
    
    os.makedirs("results/EXP021A", exist_ok=True)
    with open("results/EXP021A/dataset_validation.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(json.dumps(results, indent=4))
    
    all_match = all([
        results["quotes_match"],
        results["candidates_match"],
        results["gold_match"],
        results["dimension_match"],
        results["tensor_match"]
    ])
    
    if all_match:
        print("Validation PASSED (100% match).")
    else:
        print("Validation FAILED.")

if __name__ == "__main__":
    main()
