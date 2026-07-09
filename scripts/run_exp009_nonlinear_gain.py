import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay

from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir
from src.models.classical_models import PointwiseLogisticRanker

setup_logging()
logger = get_logger("exp009_nonlinear_gain")

def get_ranking_metrics(y_true, y_prob, group_ids):
    df = pd.DataFrame({'group': group_ids, 'y_true': y_true, 'score': y_prob})
    
    mrr_sum = 0
    mean_rank_sum = 0
    recall_1 = 0
    recall_3 = 0
    total = 0
    
    for _, g in df.groupby('group'):
        if g['y_true'].sum() > 0:
            g = g.sort_values(by='score', ascending=False).reset_index(drop=True)
            gold_idx = g[g['y_true'] == 1].index[0]
            rank = gold_idx + 1
            
            mrr_sum += 1.0 / rank
            mean_rank_sum += rank
            if rank == 1:
                recall_1 += 1
            if rank <= 3:
                recall_3 += 1
            total += 1
            
    return {
        "MRR": mrr_sum / total,
        "Mean_Rank": mean_rank_sum / total,
        "Recall@1": recall_1 / total,
        "Recall@3": recall_3 / total
    }

def run_exp009():
    logger.info("Starting EXP009: Understanding Nonlinear Gain...")
    
    input_file = get_data_dir() / "phase2" / "candidate_features.csv"
    df = pd.read_csv(input_file)
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    all_features = [c for c in df.columns if c not in ["quote_id", "novel", "candidate", "gold_speaker", "split", "label"] and not c.startswith("symbolic_")]
    top_3 = ['candidate_is_explicit_mention', 'candidate_is_previous_speaker', 'candidate_is_recent_mention']
    
    # Train Models
    lr = PointwiseLogisticRanker(random_state=42)
    lr.fit(train_df[top_3], train_df['label'])
    lr_probs = lr.predict_proba(test_df[top_3])
    
    gbm = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    gbm.fit(train_df[all_features], train_df['label'])
    gbm_probs = gbm.predict_proba(test_df[all_features])[:, 1]
    
    test_df['lr_score'] = lr_probs
    test_df['gbm_score'] = gbm_probs
    
    # Analyze Quotes
    quote_results = []
    gbm_errors = []
    
    for quote_id, group in test_df.groupby('quote_id'):
        if group['label'].sum() > 0:
            lr_correct = group.loc[group['lr_score'].idxmax(), 'label'] == 1
            gbm_correct = group.loc[group['gbm_score'].idxmax(), 'label'] == 1
            
            category = "Both Wrong"
            if lr_correct and gbm_correct:
                category = "Both Correct"
            elif lr_correct and not gbm_correct:
                category = "LR Only"
            elif not lr_correct and gbm_correct:
                category = "GBM Only"
                
            gold_row = group[group['label'] == 1].iloc[0]
            
            quote_results.append({
                "quote_id": quote_id,
                "category": category,
                "conversation_length": gold_row['conversation_length'],
                "discourse_context_length": gold_row['discourse_context_length'],
                "num_candidates": len(group)
            })
            
            # Residual taxonomy for GBM
            if not gbm_correct:
                pred_row = group.loc[group['gbm_score'].idxmax()]
                error_type = "Unknown"
                if gold_row['candidate_is_recent_mention'] == 0 and gold_row['candidate_is_previous_speaker'] == 0 and gold_row['candidate_is_explicit_mention'] == 0:
                    error_type = "Gold has no explicit signals"
                elif gold_row['candidate_is_recent_mention'] == 1 and pred_row['candidate_is_previous_speaker'] == 1:
                    error_type = "Confused Mention for Previous Speaker"
                elif gold_row['discourse_context_length'] > 20:
                    error_type = "Long Context Narration"
                    
                gbm_errors.append({"error_type": error_type})
                
    q_df = pd.DataFrame(quote_results)
    
    # Part A: Prediction Comparison
    cat_counts = q_df['category'].value_counts().reset_index()
    cat_counts.columns = ['Category', 'Count']
    
    gbm_only_stats = {}
    if len(q_df[q_df['category'] == 'GBM Only']) > 0:
        gbm_only = q_df[q_df['category'] == 'GBM Only']
        both_correct = q_df[q_df['category'] == 'Both Correct']
        gbm_only_stats = {
            "Avg Conv Length (GBM Only)": gbm_only['conversation_length'].mean(),
            "Avg Conv Length (Both Correct)": both_correct['conversation_length'].mean(),
            "Avg Context Length (GBM Only)": gbm_only['discourse_context_length'].mean(),
            "Avg Context Length (Both Correct)": both_correct['discourse_context_length'].mean(),
        }
        
    # Part B: Ranking Quality
    lr_rank_metrics = get_ranking_metrics(test_df['label'], test_df['lr_score'], test_df['quote_id'])
    gbm_rank_metrics = get_ranking_metrics(test_df['label'], test_df['gbm_score'], test_df['quote_id'])
    
    rank_df = pd.DataFrame([
        {"Model": "Logistic (Top 3)", **lr_rank_metrics},
        {"Model": "HistGBM (All 13)", **gbm_rank_metrics}
    ])
    
    # Part C: Residual Taxonomy
    err_df = pd.DataFrame(gbm_errors)
    err_counts = err_df['error_type'].value_counts().reset_index() if not err_df.empty else pd.DataFrame(columns=['Error Type', 'Count'])
    
    # Part D: PDPs
    logger.info("Generating PDPs...")
    EXP_DIR = get_reports_dir() / "EXP009"
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Need to find index of features
    prev_spk_idx = all_features.index('candidate_is_previous_speaker')
    ctx_len_idx = all_features.index('discourse_context_length')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        gbm, train_df[all_features].sample(10000, random_state=42), 
        [(prev_spk_idx, ctx_len_idx)],
        ax=ax,
        grid_resolution=20
    )
    plt.title("PDP: Previous Speaker x Context Length")
    plt.tight_layout()
    plt.savefig(EXP_DIR / "pdp_interaction.png")
    plt.close()
    
    # Write Report
    report_file = EXP_DIR / "understanding_nonlinear_gain_report.md"
    with open(report_file, 'w') as f:
        f.write("# EXP009: Understanding Nonlinear Gain\n\n")
        
        f.write("## Part A: Prediction Comparison\n\n")
        f.write(cat_counts.to_markdown(index=False) + "\n\n")
        if gbm_only_stats:
            f.write("**Characteristics of GBM-Only Successes:**\n")
            for k, v in gbm_only_stats.items():
                f.write(f"- {k}: {v:.2f}\n")
            f.write("\n")
            
        f.write("## Part B: Ranking Quality\n\n")
        f.write(rank_df.to_markdown(index=False) + "\n\n")
        
        f.write("## Part C: Residual Error Taxonomy (HistGBM)\n\n")
        f.write("Did GBM solve the semantic errors? (Hint: No, if 'Gold has no explicit signals' is still the largest class).\n\n")
        f.write(err_counts.to_markdown(index=False) + "\n\n")
        
        f.write("## Part D: Targeted Interaction Interpretation\n\n")
        f.write("![PDP Interaction](/home/Aditya/speaker-attribution-research/results/EXP009/pdp_interaction.png)\n")
        
    logger.info(f"Report saved to {report_file}")

if __name__ == "__main__":
    run_exp009()
