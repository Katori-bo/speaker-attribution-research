import time
import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir
from src.evaluation.metrics import expected_calibration_error
from src.models.classical_models import PointwiseLogisticRanker

setup_logging()
logger = get_logger("exp008_nonlinear_sanity")

def evaluate_model(model_name, y_true, y_prob, group_df, train_time, eval_time):
    # Log loss
    ll = log_loss(y_true, y_prob)
    # ROC-AUC
    roc = roc_auc_score(y_true, y_prob)
    # PR-AUC
    pr = average_precision_score(y_true, y_prob)
    # ECE
    ece = expected_calibration_error(y_true, y_prob)
    
    # Ranking Accuracy
    eval_df = group_df.copy()
    eval_df['score'] = y_prob
    correct = 0
    total = 0
    for _, group in eval_df.groupby('quote_id'):
        if group['label'].sum() > 0:
            if group.loc[group['score'].idxmax(), 'label'] == 1:
                correct += 1
            total += 1
            
    acc = correct / total if total > 0 else 0
    
    return {
        "Model": model_name,
        "Accuracy": acc,
        "ROC-AUC": roc,
        "PR-AUC": pr,
        "LogLoss": ll,
        "ECE": ece,
        "Train_Time_sec": train_time,
        "Eval_Time_sec": eval_time
    }

def run_exp008():
    logger.info("Starting EXP008: Nonlinear Sanity Check...")
    
    input_file = get_data_dir() / "phase2" / "candidate_features.csv"
    df = pd.read_csv(input_file)
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    all_features = [c for c in df.columns if c not in ["quote_id", "novel", "candidate", "gold_speaker", "split", "label"] and not c.startswith("symbolic_")]
    top_3 = ['candidate_is_explicit_mention', 'candidate_is_previous_speaker', 'candidate_is_recent_mention']
    
    logger.info(f"Training on {len(all_features)} continuous features...")
    
    # 1. Logistic Regression (Top 3)
    logger.info("Training Logistic Regression (Top 3)...")
    start_t = time.time()
    lr = PointwiseLogisticRanker(random_state=42)
    lr.fit(train_df[top_3], train_df['label'])
    lr_train_t = time.time() - start_t
    
    start_e = time.time()
    lr_probs = lr.predict_proba(test_df[top_3])
    lr_eval_t = time.time() - start_e
    
    res_lr = evaluate_model("Logistic Regression (Top 3)", test_df['label'], lr_probs, test_df, lr_train_t, lr_eval_t)
    
    # 2. HistGradientBoosting (All Features)
    logger.info("Training HistGradientBoosting (All Features)...")
    start_t = time.time()
    gbm = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    gbm.fit(train_df[all_features], train_df['label'])
    gbm_train_t = time.time() - start_t
    
    start_e = time.time()
    gbm_probs = gbm.predict_proba(test_df[all_features])[:, 1]
    gbm_eval_t = time.time() - start_e
    
    res_gbm = evaluate_model("HistGBM (All Features)", test_df['label'], gbm_probs, test_df, gbm_train_t, gbm_eval_t)
    
    # Comparison
    results = pd.DataFrame([res_lr, res_gbm])
    
    EXP_DIR = get_reports_dir() / "EXP008"
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Feature Importance (Permutation)
    logger.info("Extracting Permutation Importance for HistGBM...")
    # Sample test set for faster permutation importance
    pi_sample = test_df.sample(n=min(len(test_df), 10000), random_state=42)
    pi_result = permutation_importance(gbm, pi_sample[all_features], pi_sample['label'], n_repeats=5, random_state=42, scoring='roc_auc')
    
    fi_df = pd.DataFrame({
        "Feature": all_features,
        "Importance_Mean": pi_result.importances_mean,
        "Importance_Std": pi_result.importances_std
    }).sort_values(by="Importance_Mean", ascending=False)
    
    # Write Report
    report_file = EXP_DIR / "nonlinear_sanity_check_report.md"
    with open(report_file, 'w') as f:
        f.write("# EXP008: Nonlinear Sanity Check Report\n\n")
        f.write("Does a lightweight nonlinear learner extract meaningful additional signal from the explicit representation?\n\n")
        f.write("## 1. Performance Comparison\n\n")
        f.write(results.to_markdown(index=False) + "\n\n")
        
        # Calculate Delta
        d_acc = res_gbm["Accuracy"] - res_lr["Accuracy"]
        d_roc = res_gbm["ROC-AUC"] - res_lr["ROC-AUC"]
        d_ll = res_lr["LogLoss"] - res_gbm["LogLoss"] # Note inverted for LL (lower is better)
        
        f.write(f"- **Δ Accuracy:** {d_acc*100:.3f}%\n")
        f.write(f"- **Δ ROC-AUC:** {d_roc:.4f}\n")
        f.write(f"- **Δ LogLoss:** {d_ll:.4f}\n\n")
        
        f.write("## 2. Feature Importance (HistGBM)\n\n")
        f.write("Permutation importance (scoring=roc_auc) on a holdout sample:\n\n")
        f.write(fi_df.to_markdown(index=False) + "\n\n")
        
    logger.info(f"Report saved to {report_file}")

if __name__ == "__main__":
    run_exp008()
