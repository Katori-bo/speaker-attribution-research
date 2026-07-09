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
from src.neural.models import SpeakerGRU
from scripts.run_EXP021A_2_mlp_ce import compute_metrics, bootstrap_ci

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_gru(model, dataloader, device, type_mappings, ablate_memory=False, ablate_feedback=False, ablate_shuffle=False):
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch.candidate_features.to(device)
            mask = batch.candidate_mask.to(device)
            cids = batch.candidate_ids.to(device)
            gold_index = batch.gold_index.to(device)
            q_ids = batch.quote_ids
            
            # Predict autoregressively
            scores = model(
                features, cids, mask, 
                speaker_ids_for_update=None, # Autoregressive
                ablate_memory=ablate_memory,
                ablate_feedback=ablate_feedback,
                ablate_shuffle=ablate_shuffle
            ).squeeze(0) # [seq_len, max_cand]
            
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
                
                results.append({
                    'quote_id': q_ids[i],
                    'quote_type': type_mappings.get(q_ids[i], 'Unknown'),
                    'pred_rank': rank,
                    'loss': losses[i]
                })
                
    return pd.DataFrame(results)

def main():
    os.makedirs("results/EXP021B", exist_ok=True)
    
    with open("configs/EXP021_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(config['seed'])
    device = get_device() if config['device'] == 'auto' else config['device']
    logger.info(f"Using device: {device}")
    
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
    train_seq = TensorSequenceDataset(train_df, feature_cols, feature_mode='state_free', vocab=vocab, scaler=scaler)
    test_seq = TensorSequenceDataset(test_df, feature_cols, feature_mode='state_free', vocab=vocab, scaler=scaler)
    
    # Custom collate function to avoid standard tensor stacking since batch items are objects
    def collate_fn(batch):
        return batch[0]
        
    train_loader = DataLoader(train_seq, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_seq, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    input_dim = train_seq[0].candidate_features.shape[-1]
    model = SpeakerGRU(
        feature_dim=input_dim, 
        vocab_size=len(vocab),
        emb_dim=32,
        hidden_dim=64
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    epochs = config['epochs']
    if os.environ.get("CPU_TEST_RUN") == "1":
        epochs = 1
        
    logger.info("Training Speaker-Feedback GRU...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_items = 0
        
        for batch in train_loader:
            # Add batch dimension since batch size is 1
            features = batch.candidate_features.unsqueeze(0).to(device)
            mask = batch.candidate_mask.unsqueeze(0).to(device)
            cids = batch.candidate_ids.unsqueeze(0).to(device)
            gold_speaker_ids = batch.gold_speaker_id.unsqueeze(0).to(device)
            gold_index = batch.gold_index.to(device) # already [seq_len]
            
            optimizer.zero_grad()
            
            # Teacher forcing using gold speaker IDs
            scores = model(features, cids, mask, speaker_ids_for_update=gold_speaker_ids).squeeze(0) # [seq_len, max_cand]
            loss = criterion(scores, gold_index)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(gold_index)
            total_items += len(gold_index)
            
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/total_items:.4f}")
        
    logger.info("Evaluating Speaker-Feedback GRU...")
    
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
    base_preds = evaluate_gru(model, test_loader, device, type_mappings, ablate_memory=False, ablate_feedback=False)
    base_metrics = compute_metrics(base_preds)
    
    # 2. Ablation: Reset hidden state
    logger.info("Running Ablation: Reset Hidden State...")
    reset_preds = evaluate_gru(model, test_loader, device, type_mappings, ablate_memory=True, ablate_feedback=False)
    reset_metrics = compute_metrics(reset_preds)
    
    # 3. Ablation: No Speaker Feedback
    logger.info("Running Ablation: No Speaker Feedback...")
    nofb_preds = evaluate_gru(model, test_loader, device, type_mappings, ablate_memory=False, ablate_feedback=True)
    nofb_metrics = compute_metrics(nofb_preds)
    
    # 4. Ablation: Shuffled Speaker Feedback
    logger.info("Running Ablation: Shuffled Speaker Feedback...")
    shuffle_preds = evaluate_gru(model, test_loader, device, type_mappings, ablate_memory=False, ablate_feedback=False, ablate_shuffle=True)
    shuffle_metrics = compute_metrics(shuffle_preds)
    
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
    
    # Train MLP CE quickly to get McNemar baseline
    logger.info("Training MLP CE for McNemar baseline...")
    from scripts.run_EXP021A_2_mlp_ce import QuoteTensorDataset, RankingMLP, mcnemar_test
    mlp_train_dataset = QuoteTensorDataset(train_seq)
    mlp_test_dataset = QuoteTensorDataset(test_seq)
    mlp_train_loader = DataLoader(mlp_train_dataset, batch_size=config.get('batch_size', 32), shuffle=True)
    mlp_test_loader = DataLoader(mlp_test_dataset, batch_size=256, shuffle=False)
    
    mlp_model = RankingMLP(input_dim=input_dim).to(device)
    mlp_opt = torch.optim.Adam(mlp_model.parameters(), lr=config['learning_rate'])
    for _ in range(epochs):
        mlp_model.train()
        for batch in mlp_train_loader:
            features = batch['features'].to(device)
            mask = batch['mask'].to(device)
            gold_index = batch['gold_index'].to(device)
            mlp_opt.zero_grad()
            scores = mlp_model(features, mask)
            loss = nn.CrossEntropyLoss()(scores, gold_index)
            loss.backward()
            mlp_opt.step()
            
    mlp_model.eval()
    mlp_results = []
    with torch.no_grad():
        for batch in mlp_test_loader:
            scores = mlp_model(batch['features'].to(device), batch['mask'].to(device))
            sorted_indices = torch.argsort(scores, dim=-1, descending=True)
            for i in range(len(batch['quote_id'])):
                gold = batch['gold_index'][i].item()
                ranks = (sorted_indices[i] == gold).nonzero(as_tuple=True)[0]
                rank = ranks[0].item() + 1 if len(ranks) > 0 else 999
                mlp_results.append({'quote_id': batch['quote_id'][i], 'pred_rank': rank})
    mlp_preds = pd.DataFrame(mlp_results)
    
    p_value = mcnemar_test(base_preds, mlp_preds)
    
    report = ["# EXP021B Speaker-Feedback GRU Results\n"]
    
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
    
    report.append("## 3. Feedback Ablation (Speaker vectors = Zeros)")
    report.append(f"**Overall Accuracy**: {nofb_metrics['Accuracy']*100:.2f}%")
    report.append(f"**Implicit Accuracy**: {nofb_metrics['Implicit_Accuracy']*100:.2f}%")
    report.append(f"**Anaphoric Accuracy**: {nofb_metrics['Anaphoric_Accuracy']*100:.2f}%\n")
    
    report.append("## 4. Feedback Ablation (Shuffled speaker vectors)")
    report.append(f"**Overall Accuracy**: {shuffle_metrics['Accuracy']*100:.2f}%")
    report.append(f"**Implicit Accuracy**: {shuffle_metrics['Implicit_Accuracy']*100:.2f}%")
    report.append(f"**Anaphoric Accuracy**: {shuffle_metrics['Anaphoric_Accuracy']*100:.2f}%\n")
    
    report.append("## Analysis")
    report.append(f"- McNemar p-value vs MLP CE Baseline: {p_value:.4e}")
    diff = (base_metrics['Accuracy'] - 0.6888) * 100
    report.append(f"- The GRU achieved a {'gain' if diff > 0 else 'loss'} of {abs(diff):.2f} pp against the memory-free neural baseline.")
    
    mem_diff = (base_metrics['Accuracy'] - reset_metrics['Accuracy']) * 100
    report.append(f"- The memory reset ablation caused a {'drop' if mem_diff > 0 else 'rise'} of {abs(mem_diff):.2f} pp. This isolates the exact contribution of the GRU's recurrence.")
    
    with open("results/EXP021B/metrics_report.md", "w") as f:
        f.write("\n".join(report))
        
    logger.info("EXP021B Evaluation complete. Report saved.")

if __name__ == "__main__":
    main()
