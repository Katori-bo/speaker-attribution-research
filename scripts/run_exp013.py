import os
import json
import time
import tracemalloc
import pandas as pd
import logging
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, log_loss
from sklearn.inspection import permutation_importance

from src.addressee.pipeline import AddresseeFeatureProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    "features": {
        "coreference": True,
        "addressee": True
    }
}

def get_data_dir() -> Path:
    return Path("data")

def get_ranking_metrics(y_true, y_score, groups):
    df = pd.DataFrame({'label': y_true, 'score': y_score, 'group': groups})
    df['rank'] = df.groupby('group')['score'].rank(ascending=False, method='first')
    top_preds = df[df['rank'] == 1]
    accuracy = top_preds['label'].mean()
    return accuracy

def run_exp013():
    logger.info("Starting EXP013: Speaker-Addressee Feature Evaluation")
    
    exp012_cache_file = get_data_dir() / "raw" / "pdnc" / "phase2" / "candidate_features_exp012.csv"
    if not exp012_cache_file.exists():
        logger.error(f"Cannot find EXP012 cached features at {exp012_cache_file}.")
        return
        
    df = pd.read_csv(exp012_cache_file)
    
    addressee_provider = AddresseeFeatureProvider(enabled=CONFIG["features"]["addressee"])
    novel_features_list = []
    
    start_time = time.time()
    tracemalloc.start()
    
    for novel, novel_df in df.groupby('novel'):
        addressee_provider.reset_state()
        unique_quotes = novel_df['quote_id'].unique()
        def get_q_idx(q_id):
            try: return int(q_id.split('_')[-1])
            except: return 0
        unique_quotes = sorted(unique_quotes, key=get_q_idx)
        
        for q_id in unique_quotes:
            q_idx = get_q_idx(q_id)
            q_df = novel_df[novel_df['quote_id'] == q_id]
            for _, row in q_df.iterrows():
                candidate = row['candidate']
                feats = addressee_provider.extract(novel, q_idx, candidate)
                feats['quote_id'] = q_id
                feats['candidate'] = candidate
                novel_features_list.append(feats)
            addressee_provider.update_state(novel, q_idx)
            
    addr_feat_df = pd.DataFrame(novel_features_list)
    df = df.merge(addr_feat_df, on=['quote_id', 'candidate'], how='left')
    
    extraction_time = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    base_feats = [c for c in df.columns if c not in [
        "quote_id", "novel", "candidate", "gold_speaker", "split", "label",
        "quote_start_byte", "quote_end_byte", "quoteByteSpans",
        "candidate_was_addressed", "addressee_recency", "speaker_addressee_transition",
        "candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency"
    ] and not c.startswith("symbolic_")]
    
    if CONFIG["features"]["coreference"]:
        base_feats += ["candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency"]
        
    exp_feats = base_feats.copy()
    if CONFIG["features"]["addressee"]:
        exp_feats += ["candidate_was_addressed", "addressee_recency", "speaker_addressee_transition"]
        
    # Baseline Model (EXP012B)
    baseline_model = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    baseline_model.fit(train_df[base_feats], train_df['label'])
    test_df['baseline_score'] = baseline_model.predict_proba(test_df[base_feats])[:, 1]
    
    # Experimental Model (EXP013)
    exp_model = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    exp_model.fit(train_df[exp_feats], train_df['label'])
    test_df['exp_score'] = exp_model.predict_proba(test_df[exp_feats])[:, 1]
    
    # Ranks
    test_df['baseline_rank'] = test_df.groupby('quote_id')['baseline_score'].rank(ascending=False, method='first')
    test_df['exp_rank'] = test_df.groupby('quote_id')['exp_score'].rank(ascending=False, method='first')
    
    # Metrics
    def compute_metrics(y_true, y_score, groups, ranks):
        top_preds = ranks == 1
        return {
            "Accuracy": y_true[top_preds].mean(),
            "Precision": precision_score(y_true, (y_score > 0.5).astype(int)),
            "Recall": recall_score(y_true, (y_score > 0.5).astype(int)),
            "F1": f1_score(y_true, (y_score > 0.5).astype(int)),
            "PR-AUC": average_precision_score(y_true, y_score),
            "LogLoss": log_loss(y_true, y_score)
        }
    
    baseline_metrics = compute_metrics(test_df['label'].values, test_df['baseline_score'].values, test_df['quote_id'].values, test_df['baseline_rank'].values)
    exp_metrics = compute_metrics(test_df['label'].values, test_df['exp_score'].values, test_df['quote_id'].values, test_df['exp_rank'].values)
    
    os.makedirs("results/EXP013", exist_ok=True)
    
    with open("results/EXP013/metrics.json", "w") as f:
        json.dump({
            "Baseline": baseline_metrics,
            "Experimental": exp_metrics,
            "Runtime_Seconds": extraction_time,
            "Peak_Memory_MB": peak_mem / 10**6
        }, f, indent=4)
        
    # Permutation Importance
    logger.info("Computing Permutation Importance...")
    pi = permutation_importance(exp_model, test_df[exp_feats], test_df['label'], n_repeats=5, random_state=42, scoring='roc_auc')
    importances = {exp_feats[i]: pi.importances_mean[i] for i in range(len(exp_feats))}
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    with open("results/EXP013/feature_importance.json", "w") as f:
        json.dump(sorted_importances, f, indent=4)
        
    # Oracle / Recovery Analysis
    detailed_results = []
    a_correct = b_recovered = c_regression = d_wrong = 0
    exp012_errors = 0
    
    for q_id, q_df in test_df.groupby('quote_id'):
        gold_row = q_df[q_df['label'] == 1]
        if len(gold_row) == 0: continue
        gold_row = gold_row.iloc[0]
        
        b_rank = gold_row['baseline_rank']
        e_rank = gold_row['exp_rank']
        feature_active = (gold_row['candidate_was_addressed'] == 1.0 or gold_row['speaker_addressee_transition'] == 1.0)
        
        if b_rank > 1: exp012_errors += 1
            
        if b_rank == 1 and e_rank == 1:
            cat = "A. Both correct"
            a_correct += 1
        elif b_rank > 1 and e_rank == 1:
            cat = "B. Newly recovered"
            b_recovered += 1
        elif b_rank == 1 and e_rank > 1:
            cat = "C. Regression"
            c_regression += 1
        else:
            cat = "D. Both wrong"
            d_wrong += 1
            
        detailed_results.append({
            "quote_id": q_id,
            "novel": gold_row['novel'],
            "gold_candidate": gold_row['candidate'],
            "baseline_rank": b_rank,
            "exp_rank": e_rank,
            "category": cat,
            "addressee_feature_active": feature_active
        })
        
    pred_df = pd.DataFrame(detailed_results)
    pred_df.to_csv("results/EXP013/prediction_comparison.csv", index=False)
    
    net_improvement = b_recovered - c_regression
    relative_error_reduction = (b_recovered - c_regression) / exp012_errors if exp012_errors > 0 else 0.0
    
    # Pragmatics analysis
    recovered = pred_df[pred_df['category'] == 'B. Newly recovered']
    active_recovered = recovered[recovered['addressee_feature_active'] == True]
    
    with open("results/EXP013/recovery_analysis.md", "w") as f:
        f.write("# EXP013 Recovery Analysis\n\n")
        f.write(f"## Overall Movement\n")
        f.write(f"- A. Both correct: {a_correct}\n")
        f.write(f"- B. Newly recovered: {b_recovered}\n")
        f.write(f"- C. Regression: {c_regression}\n")
        f.write(f"- D. Both wrong: {d_wrong}\n\n")
        f.write(f"**Net Improvement (B - C)**: {net_improvement} quotes\n")
        f.write(f"**Relative Error Reduction**: {relative_error_reduction:.4f}\n\n")
        f.write(f"## Pragmatics/Addressee Feature Impact\n")
        f.write(f"Of the {b_recovered} newly recovered quotes, {len(active_recovered)} had active addressee features.\n")
        
    # Write final results summary
    with open("results/EXP013/EXP013_RESULTS.md", "w") as f:
        f.write("# EXP013 Final Results\n\n")
        f.write(f"Baseline Accuracy (EXP012B): {baseline_metrics['Accuracy']:.4f}\n")
        f.write(f"Experimental Accuracy (EXP013): {exp_metrics['Accuracy']:.4f}\n")
        f.write(f"Absolute Gain: {exp_metrics['Accuracy'] - baseline_metrics['Accuracy']:.4f}\n\n")
        f.write(f"Net Quote Improvement: {net_improvement}\n")
        f.write(f"Relative Error Reduction: {relative_error_reduction:.4f}\n")

if __name__ == "__main__":
    run_exp013()
