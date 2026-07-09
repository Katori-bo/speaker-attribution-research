import os
import sys
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
from src.evaluation.runner import load_frozen_exp014_dataset
from src.neural.sequence_dataset import build_character_vocab, TensorSequenceDataset
# Import MLP and compute_metrics from EXP021A.2
from scripts.run_EXP021A_2_mlp_ce import QuoteTensorDataset, RankingMLP, compute_metrics, bootstrap_ci

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_corruption(features, mask, feature_indices, corruption_probs, device):
    batch_size, max_cand, _ = features.shape
    corrupted_features = features.clone()
    
    for feat_name, prob in corruption_probs.items():
        if feat_name not in feature_indices:
            continue
            
        f_idx = feature_indices[feat_name]
        corrupt_mask = torch.rand(batch_size, device=device) < prob
        
        # Determine if it's a quote-level feature or candidate-level
        # Quote-level features have the same value for all valid candidates in a quote
        # We can check the first item in the batch
        # But to be safe and robust, we can just shuffle values across candidates if they differ,
        # otherwise we shuffle across the batch.
        
        for b in range(batch_size):
            if corrupt_mask[b]:
                num_valid = mask[b].sum().item()
                if num_valid > 1:
                    # Shuffle among candidates for this quote
                    vals = corrupted_features[b, :num_valid, f_idx].clone()
                    if len(torch.unique(vals)) > 1:
                        shuffled_indices = torch.randperm(num_valid, device=device)
                        corrupted_features[b, :num_valid, f_idx] = vals[shuffled_indices]
                    else:
                        # It's a quote-level feature (all candidates have the same value)
                        # We simulate corruption by swapping with a random other quote in the batch
                        rand_b = torch.randint(0, batch_size, (1,)).item()
                        rand_num_valid = mask[rand_b].sum().item()
                        if rand_num_valid > 0:
                            corrupted_features[b, :num_valid, f_idx] = corrupted_features[rand_b, 0, f_idx]
                            
    return corrupted_features

def main():
    os.makedirs("results/EXP021A_3", exist_ok=True)
    
    with open("configs/EXP021_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(config['seed'])
    device = get_device() if config['device'] == 'auto' else config['device']
    logger.info(f"Using device: {device}")
    
    df = load_frozen_exp014_dataset()
    vocab = build_character_vocab(df, vocab_path="results/EXP021A_3/character_vocab.json")
    
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
    
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    
    logger.info("Building Sequence Datasets (full mode)...")
    train_seq = TensorSequenceDataset(train_df, feature_cols, feature_mode='full', vocab=vocab, scaler=scaler)
    test_seq = TensorSequenceDataset(test_df, feature_cols, feature_mode='full', vocab=vocab, scaler=scaler)
    
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
        
    feature_indices = {feat: idx for idx, feat in enumerate(feature_cols)}
    corruption_probs = {
        'candidate_is_last_speaker': 0.1575,
        'candidate_is_previous_speaker': 0.1540,
        'candidate_in_participant_stack': 0.1985,
        'candidate_stack_depth': 0.4199,
        'conversation_speaker_change': 0.3362,
        'conv_active_id': 0.0002,
        'conv_interruption_distance': 0.0002
    }
        
    logger.info("Training Robust Ranking MLP...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            features = batch['features'].to(device)
            mask = batch['mask'].to(device)
            gold_index = batch['gold_index'].to(device)
            
            # Apply dynamic corruption
            corrupted_features = apply_corruption(features, mask, feature_indices, corruption_probs, device)
            
            optimizer.zero_grad()
            scores = model(corrupted_features, mask)
            loss = criterion(scores, gold_index)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
        
    logger.info("Evaluating Robust Ranking MLP (Without Noise, mimicking TF mode)...")
    model.eval()
    
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
    
    # We will evaluate in TF mode (no noise) first to see its peak potential
    results_tf = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            mask = batch['mask'].to(device)
            gold_index = batch['gold_index'].to(device)
            q_ids = batch['quote_id']
            
            scores = model(features, mask)
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fn(scores, gold_index).cpu().numpy()
            
            sorted_indices = torch.argsort(scores, dim=-1, descending=True)
            for i in range(len(q_ids)):
                gold = gold_index[i].item()
                ranks = (sorted_indices[i] == gold).nonzero(as_tuple=True)[0]
                rank = ranks[0].item() + 1 if len(ranks) > 0 else 999
                
                results_tf.append({
                    'quote_id': q_ids[i],
                    'quote_type': type_mappings.get(q_ids[i], 'Unknown'),
                    'pred_rank': rank,
                    'loss': losses[i]
                })
                
    mlp_preds_tf = pd.DataFrame(results_tf)
    tf_metrics = compute_metrics(mlp_preds_tf)
    
    # Now simulate True AR mode evaluation by applying noise deterministically?
    # No, we must run the MLP through run_evaluation to get the true AR dynamic features!
    logger.info("Evaluating Robust Ranking MLP in True AR mode...")
    from src.evaluation.runner import run_evaluation
    from src.evaluation.discourse_mode import FullyAutoregressiveMode
    
    class RobustMLPPredictorWrapper:
        def __init__(self, model, scaler, feature_cols, device):
            self.model = model
            self.scaler = scaler
            self.feature_cols = feature_cols
            self.device = device
            self.model.eval()
            
        def predict_proba(self, df):
            X_vals = df[self.feature_cols].values
            X_scaled = self.scaler.transform(X_vals)
            # Create features tensor [1, max_candidates, feature_dim]
            features_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
            mask = torch.ones(1, len(df), dtype=torch.bool).to(self.device)
            
            with torch.no_grad():
                scores = self.model(features_t, mask).squeeze(0).cpu().numpy()
                
            out = np.zeros((len(scores), 2))
            # Softmax to get probabilities (though argmax works on raw scores)
            probs = np.exp(scores) / np.sum(np.exp(scores))
            out[:, 1] = probs
            out[:, 0] = 1 - probs
            return out
            
    mlp_wrapper = RobustMLPPredictorWrapper(model, scaler, feature_cols, device)
    fa_mode = FullyAutoregressiveMode()
    # We pass the original df (unscaled, uncorrupted) to run_evaluation.
    # run_evaluation dynamically generates AR features, calls our wrapper (which scales), and gets predictions.
    ar_preds_df = run_evaluation(fa_mode, df, mlp_wrapper, feature_cols)
    
    results_ar = []
    for q_id, q_group in ar_preds_df.groupby('quote_id'):
        q_group = q_group.sort_values('score', ascending=False).reset_index()
        gold_idx = q_group.index[q_group['candidate'] == q_group['gold_speaker']].tolist()
        rank = gold_idx[0] + 1 if len(gold_idx) > 0 else 999
        results_ar.append({
            'quote_id': q_id,
            'quote_type': type_mappings.get(q_id, 'Unknown'),
            'pred_rank': rank,
            'loss': 0.0 # LogLoss not easily computed without gold_index directly
        })
        
    mlp_preds_ar = pd.DataFrame(results_ar)
    ar_metrics = compute_metrics(mlp_preds_ar)
    
    # Compute CIs
    logger.info("Computing Bootstrapped Confidence Intervals...")
    acc_fn = lambda x: (x['pred_rank'] == 1).mean()
    imp_acc_fn = lambda x: (x[x['quote_type'] == 'Implicit']['pred_rank'] == 1).mean() if len(x[x['quote_type'] == 'Implicit']) > 0 else 0
    mrr_fn = lambda x: (1.0 / x['pred_rank']).mean()
    
    acc_ci = bootstrap_ci(mlp_preds_ar, acc_fn)
    imp_acc_ci = bootstrap_ci(mlp_preds_ar, imp_acc_fn)
    mrr_ci = bootstrap_ci(mlp_preds_ar, mrr_fn)
    
    report = ["# EXP021A.3 Robust Ranking MLP (Noise-Trained) Results\n"]
    report.append(f"**True AR Overall Accuracy**: {ar_metrics['Accuracy']*100:.2f}% (95% CI: {acc_ci[0]*100:.2f}% - {acc_ci[1]*100:.2f}%)")
    report.append(f"**True AR Implicit Accuracy**: {ar_metrics['Implicit_Accuracy']*100:.2f}% (95% CI: {imp_acc_ci[0]*100:.2f}% - {imp_acc_ci[1]*100:.2f}%)")
    report.append(f"**True AR MRR**: {ar_metrics['MRR']:.4f} (95% CI: {mrr_ci[0]:.4f} - {mrr_ci[1]:.4f})")
    report.append(f"**True AR Recall@3**: {ar_metrics['Recall@3']*100:.2f}%\n")
    
    report.append("## Sanity Checks")
    report.append(f"- TF Mode Accuracy (Training upper bound): {tf_metrics['Accuracy']*100:.2f}%")
    report.append(f"- Baseline MLP CE (State-Free): 68.88%")
    
    diff = (ar_metrics['Accuracy'] - 0.6888) * 100
    report.append(f"\n## Analysis")
    report.append(f"The Robust MLP with explicit noise training achieved {ar_metrics['Accuracy']*100:.2f}% in true Autoregressive evaluation. ")
    report.append(f"This represents a {'gain' if diff > 0 else 'loss'} of {abs(diff):.2f} pp compared to simply removing discourse features entirely (State-Free MLP CE).")
    
    if diff <= 1.0:
        report.append("\n**Conclusion**: Training with targeted feature corruption does NOT bridge the gap. Supplying broken variables is still worse or equivalent to ignoring them. The requirement for a GRU is fully validated.")
    else:
        report.append("\n**Conclusion**: Noise training successfully salvaged the discourse features, bridging the performance gap. GRU complexity may need to prove its worth against this new, robust baseline.")

    with open("results/EXP021A_3/metrics_report.md", "w") as f:
        f.write("\n".join(report))
        
    logger.info("EXP021A.3 Evaluation complete. Report saved.")

if __name__ == "__main__":
    main()
