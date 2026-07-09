import os
import sys
import json
import yaml
import torch
import logging
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.reproducibility import set_seed
from src.utils.device import get_device
from src.evaluation.runner import load_frozen_exp014_dataset, train_exp014_model, run_evaluation
from src.evaluation.discourse_mode import FullyAutoregressiveMode
from src.neural.models import CandidateMLP
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLPPredictorWrapper:
    def __init__(self, model, scaler, feature_cols, device):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.device = device
        self.model.eval()
        
    def predict_proba(self, df):
        X_vals = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X_vals)
        with torch.no_grad():
            features = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            logits = self.model(features).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            
        out = np.zeros((len(probs), 2))
        out[:, 1] = probs
        out[:, 0] = 1 - probs
        return out

def train_mlp(train_df, feature_cols, config, device, pos_weight):
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values.astype(np.float32)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = CandidateMLP(input_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    epochs = config['epochs']
    if os.environ.get("CPU_TEST_RUN") == "1":
        epochs = 1
        
    logger.info("Training MLP...")
    for epoch in range(epochs):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
    return model, scaler

def eval_tf_mlp(test_df, feature_cols, model, scaler, device):
    X_test = test_df[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    model.eval()
    all_probs = []
    with torch.no_grad():
        for (features,) in test_loader:
            features = features.to(device)
            logits = model(features).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            if probs.ndim == 0:
                all_probs.append(float(probs))
            else:
                all_probs.extend(probs.tolist())
                
    pred_df = test_df.copy()
    pred_df['score'] = all_probs
    idx_max = pred_df.groupby('quote_id')['score'].idxmax()
    return pred_df.loc[idx_max]

def eval_histgbm(df, feature_cols):
    logger.info("Training HistGBM...")
    model, _ = train_exp014_model(df)
    logger.info("Evaluating HistGBM AR...")
    fa_mode = FullyAutoregressiveMode()
    fa_preds = run_evaluation(fa_mode, df, model, feature_cols)
    idx_max = fa_preds.groupby('quote_id')['score'].idxmax()
    return fa_preds.loc[idx_max]

def main():
    with open("configs/EXP021_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(config['seed'])
    device = get_device() if config['device'] == 'auto' else config['device']
    logger.info(f"Using device: {device}")
    
    df = load_frozen_exp014_dataset()
    
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
    
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    neg_count = len(train_df) - train_df['label'].sum()
    pos_count = train_df['label'].sum()
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)
    
    # 1. Train Full MLP
    model_full, scaler_full = train_mlp(train_df, feature_cols, config, device, pos_weight)
    
    # 2. Evaluate Full MLP (Teacher Forced - frozen features)
    preds_mlp_tf = eval_tf_mlp(test_df, feature_cols, model_full, scaler_full, device)
    
    # 3. Evaluate Full MLP (Autoregressive)
    logger.info("Evaluating Full MLP AR...")
    mlp_wrapper = MLPPredictorWrapper(model_full, scaler_full, feature_cols, device)
    fa_mode = FullyAutoregressiveMode()
    ar_preds_df = run_evaluation(fa_mode, df, mlp_wrapper, feature_cols)
    idx_max_ar = ar_preds_df.groupby('quote_id')['score'].idxmax()
    preds_mlp_ar = ar_preds_df.loc[idx_max_ar]
    
    # 4. Compare AR features against TF features
    logger.info("Comparing features...")
    # ar_preds_df contains all candidates for all quotes in test_df
    # We merge on quote_id and candidate
    merged = test_df.merge(ar_preds_df, on=['quote_id', 'candidate'], suffixes=('_tf', '_ar'))
    
    dynamic_features = [
        'candidate_is_last_speaker', 'candidate_is_previous_speaker', 'candidate_is_recent_mention',
        'discourse_dialogue_position', 'conversation_turn_index', 'conversation_length',
        'conversation_speaker_change', 'conv_active_id', 'conv_interruption_distance',
        'candidate_in_participant_stack', 'candidate_stack_depth'
    ]
    
    mismatch_rates = {}
    for feat in dynamic_features:
        if f"{feat}_tf" in merged.columns and f"{feat}_ar" in merged.columns:
            mismatches = (merged[f"{feat}_tf"] != merged[f"{feat}_ar"]).sum()
            mismatch_rates[feat] = mismatches / len(merged)
            
    # 5. Train Ablated MLP
    logger.info("Training Ablated MLP...")
    ablated_feature_cols = [c for c in feature_cols if c not in dynamic_features]
    model_ablated, scaler_ablated = train_mlp(train_df, ablated_feature_cols, config, device, pos_weight)
    preds_mlp_ablated = eval_tf_mlp(test_df, ablated_feature_cols, model_ablated, scaler_ablated, device)
    
    # 6. Evaluate HistGBM AR
    preds_histgbm = eval_histgbm(df, feature_cols)
    
    # Map quote types
    q_info_dir = Path("data/raw/pdnc/data")
    type_mappings = {}
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
                    if quote_num.startswith('Q'): quote_num = quote_num[1:]
                    q_id = f"{novel}_{quote_num}"
                type_mappings[q_id] = row.get("quoteType")
                
    for pdf in [preds_mlp_tf, preds_mlp_ar, preds_mlp_ablated, preds_histgbm]:
        pdf['quote_type'] = pdf['quote_id'].map(type_mappings)
        
    def get_acc(pdf):
        return (pdf['candidate'] == pdf['gold_speaker']).mean() * 100
        
    def get_acc_by_type(pdf, qtype):
        subset = pdf[pdf['quote_type'] == qtype]
        if len(subset) == 0: return 0.0
        return (subset['candidate'] == subset['gold_speaker']).mean() * 100

    report = ["# EXP021A.1 Validation Audit Report\n"]
    
    report.append("## Test 1: Feature Equivalence Check")
    report.append("Comparing frozen features (TF) against dynamically generated Autoregressive features.\n")
    for feat, rate in mismatch_rates.items():
        report.append(f"- **{feat}** mismatch: {rate*100:.2f}%")
        
    is_leaked = any(r > 0 for r in mismatch_rates.values())
    if is_leaked:
        report.append("\n**Conclusion**: The frozen dataset does NOT match autoregressive inference. The MLP was improperly evaluated using teacher-forced (gold) states.")
    else:
        report.append("\n**Conclusion**: The frozen dataset matches autoregressive inference.")

    report.append("\n## Test 2 & 4: Model Performance Breakdown")
    report.append("Comparing models across evaluation modes (TF vs AR) and feature sets.\n")
    
    report.append("| Model | Mode | Features | Overall Acc | Explicit | Anaphoric | Implicit |")
    report.append("|-------|------|----------|-------------|----------|-----------|----------|")
    
    for name, mode, feats, pdf in [
        ("HistGBM", "AR", "All", preds_histgbm),
        ("MLP", "TF (Leaked)", "All", preds_mlp_tf),
        ("MLP", "AR", "All", preds_mlp_ar),
        ("MLP", "TF/AR", "No Discourse", preds_mlp_ablated)
    ]:
        overall = get_acc(pdf)
        explicit = get_acc_by_type(pdf, 'Explicit')
        anaphoric = get_acc_by_type(pdf, 'Anaphoric')
        implicit = get_acc_by_type(pdf, 'Implicit')
        report.append(f"| {name} | {mode} | {feats} | {overall:.2f}% | {explicit:.2f}% | {anaphoric:.2f}% | {implicit:.2f}% |")

    report.append("\n## Final Diagnosis")
    mlp_ar_implicit = get_acc_by_type(preds_mlp_ar, 'Implicit')
    histgbm_ar_implicit = get_acc_by_type(preds_histgbm, 'Implicit')
    
    if mlp_ar_implicit < 65:
        report.append("Case C: MLP AR Implicit accuracy is extremely low. The MLP is rejected as a standalone model. We must proceed to implement GRU memory to regain performance.")
    elif mlp_ar_implicit < histgbm_ar_implicit + 3:
        report.append("Case A: MLP provides comparable or slightly better performance than HistGBM in true AR mode. The 77% TF result was indeed a hallucination of gold-state leakage. We will proceed to GRU.")
    else:
        report.append("Case B: MLP AR remains remarkably high even without gold-state leakage. This suggests HistGBM was bottlenecking on noisy AR states, and the neural model is highly robust. Further MLP optimization may be needed before jumping to GRU.")

    with open("results/EXP021A_1/audit_report.md", "w") as f:
        f.write("\n".join(report))
        
    logger.info("Audit complete. Report saved to results/EXP021A_1/audit_report.md")

if __name__ == "__main__":
    main()
