import os
import sys
import json
import logging
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.runner import load_frozen_exp014_dataset
from src.neural.sequence_dataset import build_character_vocab, TensorSequenceDataset

def main():
    os.makedirs("results/EXP021B", exist_ok=True)
    df = load_frozen_exp014_dataset()
    vocab = build_character_vocab(df, vocab_path="results/EXP021B/character_vocab.json")
    
    base_feats = [c for c in df.columns if c not in [
        "quote_id", "novel", "candidate", "gold_speaker", "split", "label",
        "quote_start_byte", "quote_end_byte", "quoteByteSpans",
        "candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency",
        "candidate_is_attributed_speaker"
    ] and not c.startswith("symbolic_")]

    feature_cols = sorted(base_feats + [
        "candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency",
        "candidate_is_attributed_speaker"
    ])
    
    stats = {
        "original_quotes": len(df['quote_id'].unique()),
        "original_candidates": len(df),
        "vocab_size": len(vocab),
        "modes": {}
    }
    
    for mode in ['full', 'state_free']:
        dataset = TensorSequenceDataset(df, feature_cols, feature_mode=mode, vocab=vocab, scaler=None)
        
        mode_quotes = sum(len(seq.quote_ids) for seq in dataset)
        mode_candidates = sum(seq.candidate_mask.sum().item() for seq in dataset)
        
        stats["modes"][mode] = {
            "quotes_processed": mode_quotes,
            "candidates_processed": mode_candidates,
            "feature_dim": dataset[0].candidate_features.shape[-1] if len(dataset) > 0 else 0
        }
        
        assert mode_quotes == stats["original_quotes"], f"Quote count mismatch in {mode} mode"
        assert mode_candidates == stats["original_candidates"], f"Candidate count mismatch in {mode} mode"
        
    with open("results/EXP021B/dataset_validation.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    print("Validation successful. Results saved to results/EXP021B/dataset_validation.json")

if __name__ == "__main__":
    main()
