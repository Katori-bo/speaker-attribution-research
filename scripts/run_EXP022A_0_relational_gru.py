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
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.reproducibility import set_seed
from src.utils.device import get_device
from src.evaluation.runner import load_frozen_exp014_dataset
from src.neural.sequence_dataset import build_character_vocab, TensorSequenceDataset
from src.neural.models import RelationalSpeakerGRU
from scripts.run_EXP021A_2_mlp_ce import compute_metrics, bootstrap_ci

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_gru(model, dataloader, device, type_mappings, ablate_memory=False, ablate_shuffle=False, ablate_similarity=False):
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch.candidate_features.to(device)
            mask = batch.candidate_mask.to(device)
            gold_index = batch.gold_index.to(device)
            q_ids = batch.quote_ids
            
            # Predict autoregressively
            scores, sims = model(
                features, mask, 
                gold_index_for_update=None, # Autoregressive
                ablate_memory=ablate_memory,
                ablate_shuffle=ablate_shuffle,
                ablate_similarity=ablate_similarity
            )
            scores = scores.squeeze(0) # [seq_len, max_cand]
            sims = sims.squeeze(0) # [seq_len, max_cand, 1]
            
            # gold_index is [1, seq_len] due to DataLoader batching
            if gold_index.dim() == 2:
                gold_index = gold_index.squeeze(0)
            
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fn(scores, gold_index).cpu().numpy()
            
            sorted_indices = torch.argsort(scores, dim=-1, descending=True)
            for i in range(len(q_ids)):
                gold = gold_index[i].item()
                ranks = (sorted_indices[i] == gold).nonzero(as_tuple=True)[0]
                rank = ranks[0].item() + 1 if len(ranks) > 0 else 999
                
                pred = sorted_indices[i][0].item()
                gold_sim = sims[i, gold, 0].item()
                pred_sim = sims[i, pred, 0].item()
                
                results.append({
                    'quote_id': q_ids[i],
                    'quote_type': type_mappings.get(q_ids[i], 'Unknown'),
                    'pred_rank': rank,
                    'loss': losses[i],
                    'gold_sim': gold_sim,
                    'pred_sim': pred_sim
                })
                
    return pd.DataFrame(results)

def main():
    os.makedirs("results/EXP022A_0", exist_ok=True)
    
    with open("configs/EXP021_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(config['seed'])
    device = get_device() if config['device'] == 'auto' else config['device']
    logger.info(f"Using device: {device}")
    
    df = load_frozen_exp014_dataset()
    vocab = build_character_vocab(df, vocab_path="results/EXP022A_0/character_vocab.json")
    
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
    
    # We want state-free!
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
    
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    scaler = StandardScaler()
    scaler.fit(train_df[state_free_cols].values)
    
    logger.info("Building Sequence Datasets (state_free)...")
    logger.info(f"Using {len(state_free_cols)} state-free features: {state_free_cols}")
    train_seq = TensorSequenceDataset(train_df, state_free_cols, feature_mode='all', vocab=vocab, scaler=scaler)
    test_seq = TensorSequenceDataset(test_df, state_free_cols, feature_mode='all', vocab=vocab, scaler=scaler)
    
    for f in mutable_discourse_features:
        assert f not in train_seq.active_features
        assert f not in test_seq.active_features
    logger.info("Feature leakage assertion passed. No mutable discourse features present.")
    
    # Custom collate function to avoid standard tensor stacking since batch items are objects
    def collate_fn(batch):
        return batch[0]
        
    train_loader = DataLoader(train_seq, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_seq, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    input_dim = train_seq[0].candidate_features.shape[-1]
    model = RelationalSpeakerGRU(
        feature_dim=input_dim, 
        hidden_dim=64
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    epochs = config['epochs']
    if os.environ.get("CPU_TEST_RUN") == "1":
        epochs = 1
        
    logger.info("Training Relational Speaker GRU...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_items = 0
        
        for batch in train_loader:
            # Add batch dimension since batch size is 1
            features = batch.candidate_features.unsqueeze(0).to(device)
            mask = batch.candidate_mask.unsqueeze(0).to(device)
            gold_index_for_update = batch.gold_index.unsqueeze(0).to(device)
            gold_index = batch.gold_index.to(device) # already [seq_len]
            
            optimizer.zero_grad()
            
            # Teacher forcing using gold speaker index
            scores, _ = model(features, mask, gold_index_for_update=gold_index_for_update)
            scores = scores.squeeze(0) # [seq_len, max_cand]
            loss = criterion(scores, gold_index)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(gold_index)
            total_items += len(gold_index)
            
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/total_items:.4f}")
        
    logger.info("Evaluating Relational Speaker GRU...")
    
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
                
    # 1. Base Evaluation (Autoregressive)
    logger.info("Running Standard AR Evaluation...")
    base_preds = evaluate_gru(model, test_loader, device, type_mappings, ablate_memory=False, ablate_shuffle=False, ablate_similarity=False)
    base_metrics = compute_metrics(base_preds)
    
    # 2. Ablation: Reset hidden state
    logger.info("Running Ablation: Reset Hidden State...")
    reset_preds = evaluate_gru(model, test_loader, device, type_mappings, ablate_memory=True, ablate_shuffle=False, ablate_similarity=False)
    reset_metrics = compute_metrics(reset_preds)
    
    # 3. Ablation: Shuffled Speaker Feedback
    logger.info("Running Ablation: Shuffled Speaker Feedback...")
    shuffle_preds = evaluate_gru(model, test_loader, device, type_mappings, ablate_memory=False, ablate_shuffle=True, ablate_similarity=False)
    shuffle_metrics = compute_metrics(shuffle_preds)
    
    # 4. Ablation: Similarity Removed
    logger.info("Running Ablation: Similarity Removed...")
    nosim_preds = evaluate_gru(model, test_loader, device, type_mappings, ablate_memory=False, ablate_shuffle=False, ablate_similarity=True)
    nosim_metrics = compute_metrics(nosim_preds)
    
    # Compute Significance
    logger.info("Computing Bootstrapped Confidence Intervals...")
    acc_fn = lambda x: (x['pred_rank'] == 1).mean()
    imp_acc_fn = lambda x: (x[x['quote_type'] == 'Implicit']['pred_rank'] == 1).mean() if len(x[x['quote_type'] == 'Implicit']) > 0 else 0
    ana_acc_fn = lambda x: (x[x['quote_type'] == 'Anaphoric']['pred_rank'] == 1).mean() if len(x[x['quote_type'] == 'Anaphoric']) > 0 else 0
    mrr_fn = lambda x: (1.0 / x['pred_rank']).mean()
    
    acc_ci = bootstrap_ci(base_preds, acc_fn)
    imp_acc_ci = bootstrap_ci(base_preds, imp_acc_fn)
    ana_acc_ci = bootstrap_ci(base_preds, ana_acc_fn)
    mrr_ci = bootstrap_ci(base_preds, mrr_fn)
    
    # Load MLP CE baseline for McNemar
    logger.info("Loading MLP CE baseline for McNemar test...")
    try:
        mlp_preds = pd.read_csv("results/EXP021A_2/predictions.csv")
        from scripts.run_EXP021A_2_mlp_ce import mcnemar_test
        p_value = mcnemar_test(base_preds, mlp_preds)
    except FileNotFoundError:
        logger.warning("results/EXP021A_2/predictions.csv not found. McNemar test skipped.")
        p_value = 1.0
    
    report = ["# EXP022A.0 Relational Speaker GRU Results\n"]
    
    report.append("## 1. Full Autoregressive Evaluation")
    report.append(f"**Overall Accuracy**: {base_metrics['Accuracy']*100:.2f}% (95% CI: {acc_ci[0]*100:.2f}% - {acc_ci[1]*100:.2f}%)")
    report.append(f"**Implicit Accuracy**: {base_metrics['Implicit_Accuracy']*100:.2f}% (95% CI: {imp_acc_ci[0]*100:.2f}% - {imp_acc_ci[1]*100:.2f}%)")
    report.append(f"**Anaphoric Accuracy**: {base_metrics['Anaphoric_Accuracy']*100:.2f}% (95% CI: {ana_acc_ci[0]*100:.2f}% - {ana_acc_ci[1]*100:.2f}%)")
    report.append(f"**MRR**: {base_metrics['MRR']:.4f} (95% CI: {mrr_ci[0]:.4f} - {mrr_ci[1]:.4f})")
    report.append(f"**Recall@3**: {base_metrics['Recall@3']*100:.2f}%")
    report.append(f"**LogLoss**: {base_metrics['LogLoss']:.4f}\n")
    
    report.append("## 2. Memory Ablation (Reset state every quote)")
    report.append(f"**Overall Accuracy**: {reset_metrics['Accuracy']*100:.2f}%")
    report.append(f"**Implicit Accuracy**: {reset_metrics['Implicit_Accuracy']*100:.2f}%")
    report.append(f"**Anaphoric Accuracy**: {reset_metrics['Anaphoric_Accuracy']*100:.2f}%\n")
    
    report.append("## 3. Feedback Ablation (Shuffled speaker vectors)")
    report.append(f"**Overall Accuracy**: {shuffle_metrics['Accuracy']*100:.2f}%")
    report.append(f"**Implicit Accuracy**: {shuffle_metrics['Implicit_Accuracy']*100:.2f}%")
    report.append(f"**Anaphoric Accuracy**: {shuffle_metrics['Anaphoric_Accuracy']*100:.2f}%\n")
    
    report.append("## 4. Similarity Ablation (Cosine = 0)")
    report.append(f"**Overall Accuracy**: {nosim_metrics['Accuracy']*100:.2f}%")
    report.append(f"**Implicit Accuracy**: {nosim_metrics['Implicit_Accuracy']*100:.2f}%")
    report.append(f"**Anaphoric Accuracy**: {nosim_metrics['Anaphoric_Accuracy']*100:.2f}%\n")
    
    report.append("## 5. Similarity Diagnostics (Base Model)")
    correct_df = base_preds[base_preds['pred_rank'] == 1]
    wrong_df = base_preds[base_preds['pred_rank'] > 1]
    
    sim_correct = correct_df['gold_sim'].mean() if len(correct_df) > 0 else 0
    sim_wrong_gold = wrong_df['gold_sim'].mean() if len(wrong_df) > 0 else 0
    sim_wrong_pred = wrong_df['pred_sim'].mean() if len(wrong_df) > 0 else 0
    
    report.append(f"**Mean similarity when predicting correctly**: {sim_correct:.4f}")
    report.append(f"**Mean similarity of Gold when predicting wrongly**: {sim_wrong_gold:.4f}")
    report.append(f"**Mean similarity of Predicted when predicting wrongly**: {sim_wrong_pred:.4f}\n")
    
    report.append("## Analysis")
    report.append(f"- McNemar p-value vs MLP CE Baseline: {p_value:.4e}")
    diff = (base_metrics['Accuracy'] - 0.6888) * 100
    report.append(f"- The Relational GRU achieved a {'gain' if diff > 0 else 'loss'} of {abs(diff):.2f} pp against the memory-free neural baseline.")
    
    mem_diff = (base_metrics['Accuracy'] - reset_metrics['Accuracy']) * 100
    report.append(f"- The memory reset ablation caused a {'drop' if mem_diff > 0 else 'rise'} of {abs(mem_diff):.2f} pp. This isolates the exact contribution of the GRU's recurrence.")
    
    with open("results/EXP022A_0/metrics_report.md", "w") as f:
        f.write("\n".join(report))
        
    logger.info("EXP022A_0 Evaluation complete. Report saved.")

if __name__ == "__main__":
    main()
