import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import scipy.stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.reproducibility import set_seed
from src.utils.device import get_device
from src.evaluation.runner import load_frozen_exp014_dataset, train_exp014_model, run_evaluation
from src.evaluation.discourse_mode import FullyAutoregressiveMode
from src.neural.sequence_dataset import build_character_vocab, TensorSequenceDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuoteTensorDataset(Dataset):
    def __init__(self, seq_dataset):
        self.quotes = []
        for seq in seq_dataset:
            for q_idx in range(len(seq.quote_ids)):
                # Only include quotes that have a valid gold index among the candidates
                # A valid gold index is guaranteed by sequence_dataset.py (defaults to 0 if not found, but we should check label)
                # Actually sequence dataset doesn't filter.
                self.quotes.append({
                    'quote_id': seq.quote_ids[q_idx],
                    'features': seq.candidate_features[q_idx],
                    'mask': seq.candidate_mask[q_idx],
                    'gold_index': seq.gold_index[q_idx]
                })
                
    def __len__(self):
        return len(self.quotes)
        
    def __getitem__(self, idx):
        return self.quotes[idx]

class RankingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, mask):
        scores = self.mlp(x).squeeze(-1) 
        scores = scores.masked_fill(~mask, float('-inf'))
        return scores

def compute_metrics(preds_df):
    total = len(preds_df)
    acc = (preds_df['pred_rank'] == 1).mean()
    mrr = (1.0 / preds_df['pred_rank']).mean()
    recall3 = (preds_df['pred_rank'] <= 3).mean()
    
    implicit_df = preds_df[preds_df['quote_type'] == 'Implicit']
    implicit_acc = (implicit_df['pred_rank'] == 1).mean() if len(implicit_df) > 0 else 0
    
    # LogLoss (Cross Entropy)
    log_loss = preds_df['loss'].mean()
    
    return {
        "Accuracy": acc,
        "Implicit_Accuracy": implicit_acc,
        "MRR": mrr,
        "Recall@3": recall3,
        "LogLoss": log_loss
    }

def bootstrap_ci(preds_df, metric_fn, n_bootstraps=1000, ci=0.95):
    values = []
    n = len(preds_df)
    for _ in range(n_bootstraps):
        sample = preds_df.sample(n=n, replace=True)
        values.append(metric_fn(sample))
    
    values = np.array(values)
    lower = np.percentile(values, (1 - ci) / 2 * 100)
    upper = np.percentile(values, (1 + ci) / 2 * 100)
    return lower, upper

def mcnemar_test(preds_a, preds_b):
    # Ensure aligned by quote_id
    merged = preds_a.merge(preds_b, on='quote_id', suffixes=('_a', '_b'))
    
    n00 = ((merged['pred_rank_a'] != 1) & (merged['pred_rank_b'] != 1)).sum()
    n11 = ((merged['pred_rank_a'] == 1) & (merged['pred_rank_b'] == 1)).sum()
    n01 = ((merged['pred_rank_a'] != 1) & (merged['pred_rank_b'] == 1)).sum()
    n10 = ((merged['pred_rank_a'] == 1) & (merged['pred_rank_b'] != 1)).sum()
    
    b = n01
    c = n10
    
    if b + c == 0:
        return 1.0 # P-value 1
        
    statistic = (abs(b - c) - 1)**2 / (b + c)
    p_value = scipy.stats.chi2.sf(statistic, 1)
    return p_value

def main():
    os.makedirs("results/EXP021A_2", exist_ok=True)
    
    with open("configs/EXP021_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(config['seed'])
    device = get_device() if config['device'] == 'auto' else config['device']
    logger.info(f"Using device: {device}")
    
    df = load_frozen_exp014_dataset()
    vocab = build_character_vocab(df, vocab_path="results/EXP021A_2/character_vocab.json")
    
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
    
    mutable_discourse_features = [
        'candidate_is_last_speaker',
        'candidate_is_previous_speaker',
        'candidate_in_participant_stack',
        'candidate_stack_depth',
        'conversation_speaker_change',
        'conv_active_id',
        'conv_interruption_distance'
    ]
    state_free_cols = [c for c in feature_cols if c not in mutable_discourse_features]
    
    scaler = StandardScaler()
    scaler.fit(train_df[state_free_cols].values)
    
    logger.info("Building Sequence Datasets (state_free)...")
    train_seq = TensorSequenceDataset(train_df, feature_cols, feature_mode='state_free', vocab=vocab, scaler=scaler)
    test_seq = TensorSequenceDataset(test_df, feature_cols, feature_mode='state_free', vocab=vocab, scaler=scaler)
    
    train_dataset = QuoteTensorDataset(train_seq)
    test_dataset = QuoteTensorDataset(test_seq)
    
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    input_dim = train_seq[0].candidate_features.shape[-1]
    model = RankingMLP(input_dim=input_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    epochs = config['epochs']
    if os.environ.get("CPU_TEST_RUN") == "1":
        epochs = 1
        
    logger.info("Training Ranking MLP...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            features = batch['features'].to(device)
            mask = batch['mask'].to(device)
            gold_index = batch['gold_index'].to(device)
            
            optimizer.zero_grad()
            scores = model(features, mask)
            loss = criterion(scores, gold_index)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
        
    logger.info("Evaluating Ranking MLP...")
    model.eval()
    
    # Load quote types
    q_info_dir = Path("data/raw/pdnc/data")
    type_mappings = {}
    for novel in test_df['novel'].unique():
        q_info_path = q_info_dir / novel / "quotation_info.csv"
        if q_info_path.exists():
            q_info = pd.read_csv(q_info_path)
            for idx, row in q_info.iterrows():
                q_id_raw = row.get("quoteID")
                if pd.isna(q_id_raw) or not q_id_raw: q_id = f"{novel}_{idx}"
                else:
                    quote_num = str(q_id_raw).strip()
                    if quote_num.startswith('Q'): quote_num = quote_num[1:]
                    q_id = f"{novel}_{quote_num}"
                type_mappings[q_id] = row.get("quoteType")
    
    results = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            mask = batch['mask'].to(device)
            gold_index = batch['gold_index'].to(device)
            q_ids = batch['quote_id']
            
            scores = model(features, mask)
            
            # Compute loss per item
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fn(scores, gold_index).cpu().numpy()
            
            sorted_indices = torch.argsort(scores, dim=-1, descending=True)
            
            for i in range(len(q_ids)):
                gold = gold_index[i].item()
                ranks = (sorted_indices[i] == gold).nonzero(as_tuple=True)[0]
                rank = ranks[0].item() + 1 if len(ranks) > 0 else 999
                
                results.append({
                    'quote_id': q_ids[i],
                    'quote_type': type_mappings.get(q_ids[i], 'Unknown'),
                    'pred_rank': rank,
                    'loss': losses[i]
                })
                
    mlp_preds = pd.DataFrame(results)
    mlp_metrics = compute_metrics(mlp_preds)
    
    logger.info("Generating HistGBM AR baseline for McNemar test...")
    hist_model, _ = train_exp014_model(df)
    fa_mode = FullyAutoregressiveMode()
    fa_preds = run_evaluation(fa_mode, df, hist_model, feature_cols)
    
    # Process HistGBM AR to rank format
    histgbm_results = []
    for q_id, q_group in fa_preds.groupby('quote_id'):
        q_group = q_group.sort_values('score', ascending=False).reset_index()
        gold_idx = q_group.index[q_group['candidate'] == q_group['gold_speaker']].tolist()
        rank = gold_idx[0] + 1 if len(gold_idx) > 0 else 999
        histgbm_results.append({
            'quote_id': q_id,
            'pred_rank': rank
        })
    histgbm_preds = pd.DataFrame(histgbm_results)
    
    # Compute Significance
    logger.info("Computing Bootstrapped Confidence Intervals...")
    acc_fn = lambda x: (x['pred_rank'] == 1).mean()
    imp_acc_fn = lambda x: (x[x['quote_type'] == 'Implicit']['pred_rank'] == 1).mean() if len(x[x['quote_type'] == 'Implicit']) > 0 else 0
    mrr_fn = lambda x: (1.0 / x['pred_rank']).mean()
    
    acc_ci = bootstrap_ci(mlp_preds, acc_fn)
    imp_acc_ci = bootstrap_ci(mlp_preds, imp_acc_fn)
    mrr_ci = bootstrap_ci(mlp_preds, mrr_fn)
    
    p_value = mcnemar_test(mlp_preds, histgbm_preds)
    
    report = ["# EXP021A.2 Ranking MLP (CrossEntropy) Results\n"]
    report.append(f"**Overall Accuracy**: {mlp_metrics['Accuracy']*100:.2f}% (95% CI: {acc_ci[0]*100:.2f}% - {acc_ci[1]*100:.2f}%)")
    report.append(f"**Implicit Accuracy**: {mlp_metrics['Implicit_Accuracy']*100:.2f}% (95% CI: {imp_acc_ci[0]*100:.2f}% - {imp_acc_ci[1]*100:.2f}%)")
    report.append(f"**MRR**: {mlp_metrics['MRR']:.4f} (95% CI: {mrr_ci[0]:.4f} - {mrr_ci[1]:.4f})")
    report.append(f"**Recall@3**: {mlp_metrics['Recall@3']*100:.2f}%")
    report.append(f"**LogLoss (CrossEntropy)**: {mlp_metrics['LogLoss']:.4f}\n")
    
    report.append("## Statistical Significance (vs HistGBM AR)")
    hist_acc = (histgbm_preds['pred_rank'] == 1).mean()
    report.append(f"- HistGBM AR Accuracy: {hist_acc*100:.2f}%")
    report.append(f"- McNemar p-value: {p_value:.4e}")
    if p_value < 0.05:
        report.append("- **Conclusion**: The performance difference is statistically significant.")
    else:
        report.append("- **Conclusion**: The performance difference is NOT statistically significant.")
        
    report.append("\n## Analysis")
    report.append("This establishes the `MLP CE (state-free)` baseline. Any gains made by the GRU will be measured directly against these numbers to isolate the contribution of recurrent memory.")

    with open("results/EXP021A_2/metrics_report.md", "w") as f:
        f.write("\n".join(report))
        
    logger.info("EXP021A.2 Evaluation complete. Report saved.")

if __name__ == "__main__":
    main()
