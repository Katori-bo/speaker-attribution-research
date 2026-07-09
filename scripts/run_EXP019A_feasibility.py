import pandas as pd
import numpy as np
import json
import random
from pathlib import Path
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.evaluation.runner import load_frozen_exp014_dataset
from src.utils.logger import setup_logging, get_logger

logger = get_logger("EXP019A.0_Feasibility")

def evaluate_quote(quote_text, gold_speaker, candidates, fingerprints, eval_dict):
    eval_dict['evaluated_quotes'] += 1
    
    if gold_speaker not in candidates:
        eval_dict['skipped_gold_not_in_candidates'] += 1
        return
        
    # Candidate-restricted baselines
    # Random
    if random.choice(candidates) == gold_speaker:
        eval_dict['random_correct'] += 1
        
    # Frequency
    cand_freqs = {c: len(fingerprints.get(c, [])) for c in candidates}
    max_freq = max(cand_freqs.values()) if cand_freqs else 0
    best_freq_cands = [c for c, f in cand_freqs.items() if f == max_freq]
    if best_freq_cands and random.choice(best_freq_cands) == gold_speaker:
        eval_dict['frequency_correct'] += 1
        
    # TF-IDF Cosine Similarity
    cand_histories = []
    valid_cands_for_tfidf = []
    for c in candidates:
        if c in fingerprints and len(fingerprints[c]) > 0:
            cand_histories.append(" ".join(fingerprints[c]))
            valid_cands_for_tfidf.append(c)
            
    cand_sims = {c: 0.0 for c in candidates}
    
    if valid_cands_for_tfidf:
        corpus = [quote_text] + cand_histories
        try:
            # Note: Do not optimize parameters (lowercase=True, default tokenizer)
            vecs = TfidfVectorizer().fit_transform(corpus)
            sims = cosine_similarity(vecs[0:1], vecs[1:]).flatten()
            for i, c in enumerate(valid_cands_for_tfidf):
                cand_sims[c] = sims[i]
        except ValueError:
            pass
            
    # Rank candidates by similarity (descending), break ties randomly
    ranked_cands = sorted(candidates, key=lambda c: (cand_sims[c], random.random()), reverse=True)
    
    gold_rank = ranked_cands.index(gold_speaker) + 1
    
    if gold_rank == 1:
        eval_dict['tfidf_top1_correct'] += 1
    if gold_rank <= 3:
        eval_dict['tfidf_top3_correct'] += 1
    eval_dict['tfidf_mrr_sum'] += 1.0 / gold_rank

def compute_percentages(eval_dict):
    eval_valid = eval_dict['evaluated_quotes'] - eval_dict['skipped_gold_not_in_candidates']
    if eval_valid == 0:
        return {}
        
    return {
        "total_quotes": eval_dict['total_quotes'],
        "evaluated_quotes": eval_dict['evaluated_quotes'],
        "skipped_gold_not_in_candidates": eval_dict['skipped_gold_not_in_candidates'],
        "candidate_recall": (eval_valid / eval_dict['evaluated_quotes']) * 100 if eval_dict['evaluated_quotes'] > 0 else 0.0,
        "candidate_restricted_accuracy": (eval_dict['tfidf_top1_correct'] / eval_valid) * 100,
        "random_accuracy": (eval_dict['random_correct'] / eval_valid) * 100,
        "frequency_accuracy": (eval_dict['frequency_correct'] / eval_valid) * 100,
        "tfidf_top1": (eval_dict['tfidf_top1_correct'] / eval_valid) * 100,
        "tfidf_top3": (eval_dict['tfidf_top3_correct'] / eval_valid) * 100,
        "tfidf_mrr": (eval_dict['tfidf_mrr_sum'] / eval_valid)
    }

def main():
    setup_logging()
    logger.info("Starting EXP019A.0 Feasibility Test...")
    
    logger.info("Loading frozen EXP014 dataset...")
    df = load_frozen_exp014_dataset()
    test_df = df[df['split'] == 'test']
    
    explicit_eval = {
        "total_quotes": 0, "evaluated_quotes": 0, "skipped_gold_not_in_candidates": 0,
        "tfidf_top1_correct": 0, "tfidf_top3_correct": 0, "tfidf_mrr_sum": 0.0,
        "random_correct": 0, "frequency_correct": 0
    }
    
    implicit_eval = {
        "total_quotes": 0, "evaluated_quotes": 0, "skipped_gold_not_in_candidates": 0,
        "tfidf_top1_correct": 0, "tfidf_top3_correct": 0, "tfidf_mrr_sum": 0.0,
        "random_correct": 0, "frequency_correct": 0
    }
    
    characters_with_fingerprints = 0
    
    novels = test_df['novel'].unique()
    q_info_dir = "data/raw/pdnc/data"
    
    for novel in novels:
        logger.info(f"Processing {novel}...")
        q_info_path = os.path.join(q_info_dir, novel, "quotation_info.csv")
        q_info = pd.read_csv(q_info_path)
        
        fingerprints = {} # character -> list of explicitly attributed texts
        
        for idx, row in q_info.iterrows():
            q_type = str(row.get('quoteType', ''))
            q_id_raw = row.get("quoteID")
            if pd.isna(q_id_raw) or not q_id_raw:
                q_id = f"{novel}_{idx}"
            else:
                quote_num = str(q_id_raw).strip()
                if quote_num.startswith('Q'):
                    quote_num = quote_num[1:]
                q_id = f"{novel}_{quote_num}"
                
            gold_speaker = str(row.get('speaker', 'Unknown')).strip()
            quote_text = str(row.get('quoteText', ''))
            
            candidates_df = test_df[test_df['quote_id'] == q_id]
            candidates = candidates_df['candidate'].tolist() if not candidates_df.empty else []
            
            if q_type == 'Explicit':
                explicit_eval['total_quotes'] += 1
                if candidates:
                    evaluate_quote(quote_text, gold_speaker, candidates, fingerprints, explicit_eval)
                
                # Update fingerprint for Explicit quotes
                if gold_speaker not in fingerprints:
                    fingerprints[gold_speaker] = []
                fingerprints[gold_speaker].append(quote_text)
                
            elif q_type == 'Implicit':
                implicit_eval['total_quotes'] += 1
                if candidates:
                    evaluate_quote(quote_text, gold_speaker, candidates, fingerprints, implicit_eval)

        characters_with_fingerprints += len(fingerprints)

    out_data = {
        "explicit_eval": compute_percentages(explicit_eval),
        "implicit_eval": compute_percentages(implicit_eval),
        "characters_with_fingerprints": characters_with_fingerprints
    }
    
    out_dir = Path("results/EXP019A")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "feasibility.json"
    
    with open(out_file, 'w') as f:
        json.dump(out_data, f, indent=4)
        
    logger.info("Feasibility Results:")
    logger.info(json.dumps(out_data, indent=4))
    logger.info(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
