import os
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.inspection import permutation_importance
from src.evaluation.runner import load_frozen_exp014_dataset, train_exp014_model, run_evaluation
from src.evaluation.discourse_mode import FullyAutoregressiveMode
from src.style.pipeline import StyleFeatureProvider
from src.utils.logger import setup_logging, get_logger

logger = get_logger("EXP019A_Runner")

def compute_train_lexical_similarity(df, min_quotes=5):
    """
    Computes character_lexical_similarity for the training set.
    """
    logger.info("Computing character_lexical_similarity for the training set...")
    df = df.copy()
    
    # Accumulate updates in a dictionary to assign in bulk
    similarity_updates = {}
    
    # Pre-group training candidates by (novel, quote_id) for instant O(1) lookups
    train_df = df[df['split'] == 'train']
    candidates_by_quote = train_df.groupby(['novel', 'quote_id'])['candidate'].apply(list).to_dict()
    
    q_info_dir = Path("data/raw/pdnc/data")
    
    for novel in df['novel'].unique():
        q_info_path = q_info_dir / novel / "quotation_info.csv"
        if not q_info_path.exists():
            continue
        q_info = pd.read_csv(q_info_path)
        
        style_provider = StyleFeatureProvider(min_quotes=min_quotes)
        
        for idx, row in q_info.iterrows():
            q_id_raw = row.get("quoteID")
            if pd.isna(q_id_raw) or not q_id_raw:
                q_id = f"{novel}_{idx}"
            else:
                quote_num = str(q_id_raw).strip()
                if quote_num.startswith('Q'):
                    quote_num = quote_num[1:]
                q_id = f"{novel}_{quote_num}"
                
            quote_text = str(row.get("quoteText", ""))
            gold_speaker = str(row.get("speaker", "Unknown")).strip()
            
            candidates = candidates_by_quote.get((novel, q_id))
            if candidates:
                scores = style_provider.extract_features(quote_text, candidates)
                for c in candidates:
                    similarity_updates[(novel, q_id, c)] = scores[c]
            
            if row.get("quoteType") == "Explicit" and gold_speaker != "Unknown":
                style_provider.update_state(gold_speaker, quote_text)
                
    # Assign in bulk using fast itertuples lookup, preserving original index and row ordering
    df['character_lexical_similarity'] = [
        similarity_updates.get((r.novel, r.quote_id, r.candidate), 0.0)
        for r in df.itertuples()
    ]
    
    return df

def calculate_accuracy(predictions_df):
    """Calculates overall and implicit accuracy from predictions."""
    # Deduplicate at quote level by taking prediction with highest score
    quote_preds = []
    for q_id, q_df in predictions_df.groupby('quote_id'):
        best_row = q_df.loc[q_df['score'].idxmax()]
        # Check correctness
        correct = 1 if best_row['candidate'] == best_row['gold_speaker'] else 0
        
        # Determine quote type from label or find from first row
        # (We can check candidate_is_attributed_speaker or load from q_info if needed, 
        # but in static df we don't have quoteType directly. Let's lookup quoteType).
        quote_preds.append({
            "quote_id": q_id,
            "novel": best_row['novel'],
            "correct": correct,
            "gold": best_row['gold_speaker'],
            "pred": best_row['candidate']
        })
    preds_df = pd.DataFrame(quote_preds)
    
    # We must merge with raw quoteType to separate implicit vs explicit
    q_info_dir = Path("data/raw/pdnc/data")
    type_mappings = []
    for novel in preds_df['novel'].unique():
        q_info_path = q_info_dir / novel / "quotation_info.csv"
        if q_info_path.exists():
            q_info = pd.read_csv(q_info_path)
            for idx, row in q_info.iterrows():
                q_id_raw = row.get("quoteID")
                if pd.isna(q_id_raw) or not q_id_raw:
                    q_id = f"{novel}_{idx}"
                else:
                    quote_num = str(q_id_raw).strip()
                    if quote_num.startswith('Q'):
                        quote_num = quote_num[1:]
                    q_id = f"{novel}_{quote_num}"
                type_mappings.append({"quote_id": q_id, "quote_type": row.get("quoteType")})
                
    type_df = pd.DataFrame(type_mappings)
    preds_df = preds_df.merge(type_df, on="quote_id", how="left")
    
    overall_acc = preds_df['correct'].mean() * 100
    implicit_df = preds_df[preds_df['quote_type'] == 'Implicit']
    implicit_acc = implicit_df['correct'].mean() * 100 if not implicit_df.empty else 0.0
    
    explicit_df = preds_df[preds_df['quote_type'] == 'Explicit']
    explicit_acc = explicit_df['correct'].mean() * 100 if not explicit_df.empty else 0.0
    
    return overall_acc, implicit_acc, explicit_acc, preds_df

def main():
    setup_logging()
    logger.info("Initializing EXP019A full evaluation and controls...")
    
    out_dir = Path("results/EXP019A")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading baseline EXP014 dataset...")
    df = load_frozen_exp014_dataset()
    
    # Train baseline model
    logger.info("Training baseline model (EXP014)...")
    original_model, base_feats = train_exp014_model(df)
    
    # Run Baseline (EXP014 AR)
    logger.info("Evaluating baseline system (EXP014 AR)...")
    baseline_preds = run_evaluation(FullyAutoregressiveMode(), df, original_model, base_feats)
    base_overall, base_implicit, base_explicit, base_details = calculate_accuracy(baseline_preds)
    
    # Extract features for training
    df_with_lex = compute_train_lexical_similarity(df, min_quotes=5)
    
    # Train new model
    logger.info("Training retrained model (EXP019A)...")
    exp019_model, exp019_feats = train_exp014_model(df_with_lex)
    
    # System 2: Gold Fingerprints
    logger.info("Evaluating EXP019A gold explicit fingerprints (Ceiling)...")
    gold_preds = run_evaluation(FullyAutoregressiveMode(), df_with_lex, exp019_model, exp019_feats, style_update_mode="gold")
    gold_overall, gold_implicit, gold_explicit, gold_details = calculate_accuracy(gold_preds)
    
    # System 3: Runtime Fingerprints (Real System)
    logger.info("Evaluating EXP019A runtime explicit fingerprints (Real)...")
    real_preds = run_evaluation(FullyAutoregressiveMode(), df_with_lex, exp019_model, exp019_feats, style_update_mode="real")
    real_overall, real_implicit, real_explicit, real_details = calculate_accuracy(real_preds)
    
    # System 4: Shuffled Fingerprints (Control 1: Identity Shuffle)
    logger.info("Evaluating EXP019A identity-shuffled fingerprints...")
    shuffled_preds = run_evaluation(FullyAutoregressiveMode(), df_with_lex, exp019_model, exp019_feats, 
                                    style_update_mode="real", control_mode="identity_shuffle")
    shuf_overall, shuf_implicit, shuf_explicit, shuf_details = calculate_accuracy(shuffled_preds)
    
    # Control 0: Length/Frequency Control (Frequency Shuffle)
    logger.info("Evaluating EXP019A frequency-shuffled fingerprints...")
    freq_shuffled_preds = run_evaluation(FullyAutoregressiveMode(), df_with_lex, exp019_model, exp019_feats,
                                         style_update_mode="real", control_mode="frequency_shuffle")
    freq_overall, freq_implicit, freq_explicit, freq_details = calculate_accuracy(freq_shuffled_preds)
    
    # Generate Table 1 & Table 2 Results
    results_summary = pd.DataFrame([
        {"System": "EXP014 AR Baseline", "Overall Accuracy": f"{base_overall:.2f}%", "Implicit Accuracy": f"{base_implicit:.2f}%", "Explicit Accuracy": f"{base_explicit:.2f}%"},
        {"System": "EXP019A Gold Fingerprints (Ceiling)", "Overall Accuracy": f"{gold_overall:.2f}%", "Implicit Accuracy": f"{gold_implicit:.2f}%", "Explicit Accuracy": f"{gold_explicit:.2f}%"},
        {"System": "EXP019A Runtime Fingerprints (Real)", "Overall Accuracy": f"{real_overall:.2f}%", "Implicit Accuracy": f"{real_implicit:.2f}%", "Explicit Accuracy": f"{real_explicit:.2f}%"},
        {"System": "EXP019A Frequency Shuffle (Control 0)", "Overall Accuracy": f"{freq_overall:.2f}%", "Implicit Accuracy": f"{freq_implicit:.2f}%", "Explicit Accuracy": f"{freq_explicit:.2f}%"},
        {"System": "EXP019A Identity Shuffle (Control 1)", "Overall Accuracy": f"{shuf_overall:.2f}%", "Implicit Accuracy": f"{shuf_implicit:.2f}%", "Explicit Accuracy": f"{shuf_explicit:.2f}%"}
    ])
    results_summary.to_csv(out_dir / "evaluation_summary.csv", index=False)
    logger.info("\n" + results_summary.to_string(index=False))
    
    # Control 2: Evidence Sweep
    logger.info("Running Control 2: Evidence Sweep (min_quotes sweep)...")
    sweep_results = []
    for mq in [1, 3, 5, 10, 20]:
        logger.info(f"Evaluating min_quotes = {mq}...")
        # Recalculate train features for swept min_quotes to keep model matching
        df_swept = compute_train_lexical_similarity(df, min_quotes=mq)
        model_swept, _ = train_exp014_model(df_swept)
        
        preds_swept = run_evaluation(FullyAutoregressiveMode(), df_swept, model_swept, exp019_feats, 
                                     style_update_mode="real", min_quotes=mq)
        s_overall, s_implicit, s_explicit, _ = calculate_accuracy(preds_swept)
        sweep_results.append({
            "min_quotes": mq,
            "overall_accuracy": s_overall,
            "implicit_accuracy": s_implicit,
            "explicit_accuracy": s_explicit
        })
    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv(out_dir / "min_quote_sweep.csv", index=False)
    
    # Control 3: Feature Removal Ablation
    logger.info("Running Control 3: Feature Removal Ablation...")
    ablation_preds = run_evaluation(FullyAutoregressiveMode(), df_with_lex, exp019_model, exp019_feats,
                                    style_update_mode="real", force_style_zero=True)
    ab_overall, ab_implicit, ab_explicit, _ = calculate_accuracy(ablation_preds)
    
    ablation_summary = pd.DataFrame([
        {"feature_removed": "None (Real System)", "overall_accuracy": real_overall, "implicit_accuracy": real_implicit, "logloss": 0.0}, # HistGBM LogLoss omitted or simplified
        {"feature_removed": "character_lexical_similarity", "overall_accuracy": ab_overall, "implicit_accuracy": ab_implicit, "logloss": 0.0}
    ])
    ablation_summary.to_csv(out_dir / "feature_ablation.csv", index=False)
    
    # Generate Artifact 1: character_fingerprint_distribution.csv
    logger.info("Generating character_fingerprint_distribution.csv...")
    dist_rows = []
    q_info_dir = Path("data/raw/pdnc/data")
    test_df = df[df['split'] == 'test']
    for novel in test_df['novel'].unique():
        q_info_path = q_info_dir / novel / "quotation_info.csv"
        if q_info_path.exists():
            q_info = pd.read_csv(q_info_path)
            for char, char_df in q_info[q_info['quoteType'] == 'Explicit'].groupby('speaker'):
                char_str = str(char).strip()
                quotes_count = len(char_df)
                word_count = sum(len(str(t).split()) for t in char_df['quoteText'].dropna())
                dist_rows.append({
                    "novel": novel,
                    "character": char_str,
                    "explicit_quotes_available": quotes_count,
                    "total_words": word_count
                })
    pd.DataFrame(dist_rows).to_csv(out_dir / "character_fingerprint_distribution.csv", index=False)
    
    # Generate Artifact 2: feature_importance.csv
    logger.info("Generating feature_importance.csv...")
    test_df_lex = df_with_lex[df_with_lex['split'] == 'test']
    imp_result = permutation_importance(exp019_model, test_df_lex[exp019_feats], test_df_lex['label'], random_state=42)
    importance_df = pd.DataFrame({
        "feature": exp019_feats,
        "importance": imp_result.importances_mean
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    importance_df['rank'] = importance_df.index + 1
    importance_df.to_csv(out_dir / "feature_importance.csv", index=False)
    
    # Generate Artifact 3: fingerprint_statistics.csv
    # We load statistics from StyleFeatureProvider of the Real System run
    # (Since StyleFeatureProvider is created per novel inside run_evaluation, 
    # we didn't preserve the stats directly, but we can capture stats by running a 
    # dummy logging evaluation run using the same parameters).
    logger.info("Generating fingerprint_statistics.csv...")
    dummy_provider = StyleFeatureProvider(min_quotes=5)
    for novel in test_df['novel'].unique():
        q_info_path = q_info_dir / novel / "quotation_info.csv"
        if q_info_path.exists():
            q_info = pd.read_csv(q_info_path)
            for idx, row in q_info.iterrows():
                q_id_raw = row.get("quoteID")
                if pd.isna(q_id_raw) or not q_id_raw:
                    q_id = f"{novel}_{idx}"
                else:
                    quote_num = str(q_id_raw).strip()
                    if quote_num.startswith('Q'):
                        quote_num = quote_num[1:]
                    q_id = f"{novel}_{quote_num}"
                
                quote_text = str(row.get("quoteText", ""))
                gold_speaker = str(row.get("speaker", "Unknown")).strip()
                q_type = str(row.get("quoteType", ""))
                
                q_test_rows = test_df_lex[(test_df_lex['novel'] == novel) & (test_df_lex['quote_id'] == q_id)]
                if not q_test_rows.empty:
                    candidates = q_test_rows['candidate'].tolist()
                    dummy_provider.extract_features(quote_text, candidates, quote_id=q_id, quote_type=q_type, gold_speaker=gold_speaker)
                
                if q_type == "Explicit" and gold_speaker != "Unknown":
                    dummy_provider.update_state(gold_speaker, quote_text)
                    
    pd.DataFrame(dummy_provider.stats).to_csv(out_dir / "fingerprint_statistics.csv", index=False)
    
    # Generate Artifact 4: recovery_analysis.csv
    logger.info("Generating recovery_analysis.csv...")
    recovery_df = base_details.merge(real_details, on=["quote_id", "novel", "gold"], suffixes=('_base', '_real'))
    recovery_df['recovered'] = ((recovery_df['correct_base'] == 0) & (recovery_df['correct_real'] == 1)).astype(int)
    recovery_df['regressed'] = ((recovery_df['correct_base'] == 1) & (recovery_df['correct_real'] == 0)).astype(int)
    recovery_df.rename(columns={
        "pred_base": "baseline_prediction",
        "pred_real": "EXP019_prediction",
        "quote_type_real": "quote_type"
    }, inplace=True)
    recovery_df[["quote_id", "quote_type", "baseline_prediction", "EXP019_prediction", "gold", "recovered", "regressed"]].to_csv(out_dir / "recovery_analysis.csv", index=False)
    
    # Generate Artifact 5: similarity_distribution.csv
    logger.info("Generating similarity_distribution.csv...")
    sim_rows = []
    # Using dummy_provider stats to find similarity margins
    stats_df = pd.DataFrame(dummy_provider.stats)
    for q_id, q_grp in stats_df.groupby("quote_id"):
        quote_type = q_grp.iloc[0]['quote_type']
        
        gold_row = q_grp[q_grp['is_gold'] == 1]
        wrong_rows = q_grp[q_grp['is_gold'] == 0]
        
        gold_sim = gold_row.iloc[0]['similarity_score'] if not gold_row.empty else 0.0
        max_wrong_sim = wrong_rows['similarity_score'].max() if not wrong_rows.empty else 0.0
        margin = gold_sim - max_wrong_sim
        
        # Check if correct prediction in Real System
        real_q_pred = real_details[real_details['quote_id'] == q_id]
        correct = real_q_pred.iloc[0]['correct'] if not real_q_pred.empty else 0
        
        sim_rows.append({
            "quote_id": q_id,
            "quote_type": quote_type,
            "gold_similarity": gold_sim,
            "max_wrong_similarity": max_wrong_sim,
            "similarity_margin": margin,
            "correct": correct
        })
    pd.DataFrame(sim_rows).to_csv(out_dir / "similarity_distribution.csv", index=False)
    
    # Generate Artifact 6: runtime_statistics.csv
    logger.info("Generating runtime_statistics.csv...")
    # Calculate runtime metrics
    avg_feat_time = np.mean(dummy_provider.durations) if dummy_provider.durations else 0.0
    avg_mem_per_novel = dummy_provider.get_memory_bytes() / len(test_df['novel'].unique())
    largest_fp_size = dummy_provider.get_largest_fingerprint_size()
    
    runtime_df = pd.DataFrame([{
        "average_feature_extraction_time_seconds": avg_feat_time,
        "average_fingerprint_memory_bytes": avg_mem_per_novel,
        "largest_fingerprint_size": largest_fp_size
    }])
    runtime_df.to_csv(out_dir / "runtime_statistics.csv", index=False)
    
    logger.info("All EXP019A artifacts generated successfully in results/EXP019A/.")

if __name__ == "__main__":
    main()
