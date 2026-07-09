import os
import yaml
import torch
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from src.utils.device import get_device
from src.evaluation.runner import load_frozen_exp014_dataset
from src.neural.sequence_dataset import build_character_vocab, TensorSequenceDataset
from src.neural.models import EntityAnchoredRelationalGRU
from scripts.run_EXP021A_2_mlp_ce import compute_metrics, bootstrap_ci, mcnemar_test

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_gru(model, dataloader, device, type_mappings, ablate_memory=False, 
                 ablate_shuffle=False, ablate_similarity=False, ablate_anchor_instability=False):
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch.candidate_features.to(device)
            cids = batch.candidate_ids.to(device)
            mask = batch.candidate_mask.to(device)
            gold_index = batch.gold_index.to(device)
            q_ids = batch.quote_ids
            
            # Predict autoregressively
            scores, sims = model(
                features, cids, mask, 
                gold_index_for_update=None, # Autoregressive
                ablate_memory=ablate_memory,
                ablate_shuffle=ablate_shuffle,
                ablate_similarity=ablate_similarity,
                ablate_anchor_instability=ablate_anchor_instability
            )
            scores = scores.squeeze(0) # [seq_len, max_cand]
            sims = sims.squeeze(0) # [seq_len, max_cand, 1]
            
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
    os.makedirs("results/EXP022A_1", exist_ok=True)
    
    with open("configs/EXP021_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = get_device() if config['device'] == 'auto' else config['device']
    logger.info(f"Using device: {device}")
    
    df = load_frozen_exp014_dataset()
    vocab = build_character_vocab(df, vocab_path="results/EXP022A_1/character_vocab.json")
    
    base_feats = [c for c in df.columns if c not in [
        "quote_id", "novel", "candidate", "gold_speaker", "split", "label",
        "candidate_id", "novel_candidate", "novel_gold"
    ]]
    
    mutable_discourse_features = [
        'candidate_is_last_speaker',
        'candidate_is_previous_speaker',
        'candidate_in_participant_stack',
        'candidate_stack_depth',
        'conversation_speaker_change',
        'conv_active_id',
        'conv_interruption_distance'
    ]
    
    state_free_cols = [c for c in base_feats if c not in mutable_discourse_features]
    
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    
    train_df_scaled[state_free_cols] = scaler.fit_transform(train_df[state_free_cols])
    test_df_scaled[state_free_cols] = scaler.transform(test_df[state_free_cols])
    
    logger.info("Building Sequence Datasets (state_free)...")
    logger.info(f"Using {len(state_free_cols)} state-free features: {state_free_cols}")
    train_seq = TensorSequenceDataset(train_df_scaled, state_free_cols, feature_mode='all', vocab=vocab, scaler=None)
    test_seq = TensorSequenceDataset(test_df_scaled, state_free_cols, feature_mode='all', vocab=vocab, scaler=None)
    
    for f in mutable_discourse_features:
        assert f not in train_seq.active_features
        assert f not in test_seq.active_features
    logger.info("Feature leakage assertion passed. No mutable discourse features present.")
    
    def collate_fn(batch):
        return batch[0]
        
    train_loader = DataLoader(train_seq, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_seq, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    input_dim = train_seq[0].candidate_features.shape[-1]
    
    # ----------------------------------------------------
    # Model 1: Normal Entity-Anchored Relational GRU
    # ----------------------------------------------------
    model = EntityAnchoredRelationalGRU(
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
        
    logger.info("Training Normal Entity-Anchored Relational GRU...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_items = 0
        
        for batch in train_loader:
            features = batch.candidate_features.unsqueeze(0).to(device)
            cids = batch.candidate_ids.unsqueeze(0).to(device)
            mask = batch.candidate_mask.unsqueeze(0).to(device)
            gold_index_for_update = batch.gold_index.unsqueeze(0).to(device)
            gold_index = batch.gold_index.to(device)
            
            optimizer.zero_grad()
            
            scores, _ = model(features, cids, mask, gold_index_for_update=gold_index_for_update)
            scores = scores.squeeze(0)
            loss = criterion(scores, gold_index)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(gold_index)
            total_items += len(gold_index)
            
        logger.info(f"Normal Model - Epoch {epoch+1}/{epochs} - Loss: {total_loss/total_items:.4f}")
        
    # ----------------------------------------------------
    # Model 2: No-Memory Entity Baseline
    # ----------------------------------------------------
    logger.info("Training No-Memory Entity Baseline model...")
    nomem_model = EntityAnchoredRelationalGRU(
        feature_dim=input_dim, 
        vocab_size=len(vocab),
        emb_dim=32,
        hidden_dim=64
    ).to(device)
    
    nomem_optimizer = torch.optim.Adam(nomem_model.parameters(), lr=config['learning_rate'])
    
    for epoch in range(epochs):
        nomem_model.train()
        total_loss = 0
        total_items = 0
        
        for batch in train_loader:
            features = batch.candidate_features.unsqueeze(0).to(device)
            cids = batch.candidate_ids.unsqueeze(0).to(device)
            mask = batch.candidate_mask.unsqueeze(0).to(device)
            gold_index_for_update = batch.gold_index.unsqueeze(0).to(device)
            gold_index = batch.gold_index.to(device)
            
            nomem_optimizer.zero_grad()
            
            # Train with ablate_memory=True so parameters optimize for a memory-free state
            scores, _ = nomem_model(
                features, cids, mask, 
                gold_index_for_update=gold_index_for_update,
                ablate_memory=True
            )
            scores = scores.squeeze(0)
            loss = criterion(scores, gold_index)
            
            loss.backward()
            nomem_optimizer.step()
            
            total_loss += loss.item() * len(gold_index)
            total_items += len(gold_index)
            
        logger.info(f"No-Memory Model - Epoch {epoch+1}/{epochs} - Loss: {total_loss/total_items:.4f}")
        
    logger.info("Evaluating EXP022A.1 configurations...")
    
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
                
    # 1. Normal AR Evaluation
    logger.info("Running Normal AR Evaluation...")
    base_preds = evaluate_gru(model, test_loader, device, type_mappings)
    base_metrics = compute_metrics(base_preds)
    base_preds.to_csv("results/EXP022A_1/predictions.csv", index=False)
    
    # 2. Memory Reset
    logger.info("Running Ablation: Memory Reset...")
    reset_preds = evaluate_gru(model, test_loader, device, type_mappings, ablate_memory=True)
    reset_metrics = compute_metrics(reset_preds)
    
    # 3. Shuffled Feedback
    logger.info("Running Ablation: Shuffled Feedback...")
    shuffle_preds = evaluate_gru(model, test_loader, device, type_mappings, ablate_shuffle=True)
    shuffle_metrics = compute_metrics(shuffle_preds)
    
    # 4. Anchor Instability Control
    logger.info("Running Ablation: Anchor Instability Control...")
    instable_preds = evaluate_gru(model, test_loader, device, type_mappings, ablate_anchor_instability=True)
    instable_metrics = compute_metrics(instable_preds)
    
    # 5. Similarity = 0
    logger.info("Running Ablation: Similarity = 0...")
    nosim_preds = evaluate_gru(model, test_loader, device, type_mappings, ablate_similarity=True)
    nosim_metrics = compute_metrics(nosim_preds)
    
    # 6. No-Memory Entity Baseline
    logger.info("Running No-Memory Entity Baseline Evaluation...")
    nomem_preds = evaluate_gru(nomem_model, test_loader, device, type_mappings, ablate_memory=True)
    nomem_metrics = compute_metrics(nomem_preds)
    nomem_preds.to_csv("results/EXP022A_1/nomem_predictions.csv", index=False)
    
    # Compute CIs for Normal model
    logger.info("Computing Bootstrapped Confidence Intervals...")
    acc_fn = lambda x: (x['pred_rank'] == 1).mean()
    imp_acc_fn = lambda x: (x[x['quote_type'] == 'Implicit']['pred_rank'] == 1).mean()
    ana_acc_fn = lambda x: (x[x['quote_type'] == 'Anaphoric']['pred_rank'] == 1).mean()
    mrr_fn = lambda x: (1.0 / x['pred_rank']).mean()
    
    acc_ci = bootstrap_ci(base_preds, acc_fn)
    imp_acc_ci = bootstrap_ci(base_preds, imp_acc_fn)
    ana_acc_ci = bootstrap_ci(base_preds, ana_acc_fn)
    mrr_ci = bootstrap_ci(base_preds, mrr_fn)
    
    # McNemar baseline loader
    logger.info("Loading MLP CE baseline for McNemar test...")
    mlp_preds_path = "results/EXP021A_2/predictions.csv"
    if os.path.exists(mlp_preds_path):
        mlp_preds = pd.read_csv(mlp_preds_path)
        p_val_vs_mlp = mcnemar_test(base_preds, mlp_preds)
    else:
        logger.warning("MLP predictions not found. Skipping p-val vs MLP.")
        p_val_vs_mlp = 1.0
        
    p_val_vs_nomem = mcnemar_test(base_preds, nomem_preds)
    
    report = ["# EXP022A.1 Entity-Anchored Relational GRU Results\n"]
    
    report.append("## 1. Full Autoregressive Evaluation (Normal AR)")
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
    
    report.append("## 4. Anchor Instability Control")
    report.append(f"**Overall Accuracy**: {instable_metrics['Accuracy']*100:.2f}%")
    report.append(f"**Implicit Accuracy**: {instable_metrics['Implicit_Accuracy']*100:.2f}%")
    report.append(f"**Anaphoric Accuracy**: {instable_metrics['Anaphoric_Accuracy']*100:.2f}%\n")
    
    report.append("## 5. Similarity Ablation (Cosine = 0)")
    report.append(f"**Overall Accuracy**: {nosim_metrics['Accuracy']*100:.2f}%")
    report.append(f"**Implicit Accuracy**: {nosim_metrics['Implicit_Accuracy']*100:.2f}%")
    report.append(f"**Anaphoric Accuracy**: {nosim_metrics['Anaphoric_Accuracy']*100:.2f}%\n")
    
    report.append("## 6. No-Memory Entity Baseline")
    report.append(f"**Overall Accuracy**: {nomem_metrics['Accuracy']*100:.2f}%")
    report.append(f"**Implicit Accuracy**: {nomem_metrics['Implicit_Accuracy']*100:.2f}%")
    report.append(f"**Anaphoric Accuracy**: {nomem_metrics['Anaphoric_Accuracy']*100:.2f}%\n")
    
    report.append("## 7. Similarity Diagnostics (Normal AR)")
    correct_df = base_preds[base_preds['pred_rank'] == 1]
    wrong_df = base_preds[base_preds['pred_rank'] > 1]
    
    sim_correct = correct_df['gold_sim'].mean() if len(correct_df) > 0 else 0
    sim_wrong_gold = wrong_df['gold_sim'].mean() if len(wrong_df) > 0 else 0
    sim_wrong_pred = wrong_df['pred_sim'].mean() if len(wrong_df) > 0 else 0
    
    report.append(f"**Mean similarity when predicting correctly**: {sim_correct:.4f}")
    report.append(f"**Mean similarity of Gold when predicting wrongly**: {sim_wrong_gold:.4f}")
    report.append(f"**Mean similarity of Predicted when predicting wrongly**: {sim_wrong_pred:.4f}")
    report.append(f"**Similarity Delta (Gold - Pred) when wrong**: {sim_wrong_gold - sim_wrong_pred:.4f}\n")
    
    report.append("## Analysis")
    report.append(f"- McNemar p-value vs MLP CE Baseline (68.80%): {p_val_vs_mlp:.4e}")
    report.append(f"- McNemar p-value vs No-Memory Entity Baseline: {p_val_vs_nomem:.4e}\n")
    
    diff_baseline = (base_metrics['Accuracy'] - 0.6880) * 100
    diff_nomem = (base_metrics['Accuracy'] - nomem_metrics['Accuracy']) * 100
    diff_reset = (base_metrics['Accuracy'] - reset_metrics['Accuracy']) * 100
    diff_shuffle = (base_metrics['Accuracy'] - shuffle_metrics['Accuracy']) * 100
    diff_instable = (base_metrics['Accuracy'] - instable_metrics['Accuracy']) * 100
    
    report.append(f"- **Gain vs MLP CE state-free**: {diff_baseline:.2f} pp")
    report.append(f"- **Gain vs No-Memory Entity Baseline (Memory Effect)**: {diff_nomem:.2f} pp")
    report.append(f"- **Memory Reset Ablation drop**: {diff_reset:.2f} pp")
    report.append(f"- **Shuffled Feedback Ablation drop**: {diff_shuffle:.2f} pp")
    report.append(f"- **Anchor Instability Control drop**: {diff_instable:.2f} pp")
    
    # Success Criteria Checks
    report.append("\n### Success Criteria Checks")
    report.append(f"- Normal - Reset >= 1.5 pp: {'PASS' if diff_reset >= 1.5 else 'FAIL'} ({diff_reset:.2f} pp)")
    report.append(f"- Normal - Shuffled Feedback >= 1.0 pp: {'PASS' if diff_shuffle >= 1.0 else 'FAIL'} ({diff_shuffle:.2f} pp)")
    report.append(f"- Normal - No-Memory Entity Baseline >= 1.5 pp: {'PASS' if diff_nomem >= 1.5 else 'FAIL'} ({diff_nomem:.2f} pp)")
    
    with open("results/EXP022A_1/metrics_report.md", "w") as f:
        f.write("\n".join(report))
        
    logger.info("EXP022A_1 Evaluation complete. Report saved.")

if __name__ == "__main__":
    main()
