import os
import sys
import json
import yaml
import torch
import logging
import random
import pandas as pd
import numpy as np
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Add root directory to path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.device import get_device
from src.evaluation.runner import load_frozen_exp014_dataset
from src.neural.sequence_dataset import build_character_vocab, TensorSequenceDataset
from src.neural.models import RelationalSpeakerGRU

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def batch_to_device(batch, device):
    features = batch.candidate_features
    mask = batch.candidate_mask
    if features.dim() == 3:
        features = features.unsqueeze(0)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    return (
        features.to(device),
        mask.to(device),
        batch.gold_index.to(device),
        batch.quote_ids,
        batch.candidate_ids.to(device),
        batch.gold_speaker_id.to(device)
    )

def manual_gru_cell_gates(x, h_prev, gru_cell):
    """Manually computes the GRU gates for diagnostic 5."""
    w_ih = gru_cell.weight_ih
    w_hh = gru_cell.weight_hh
    b_ih = gru_cell.bias_ih
    b_hh = gru_cell.bias_hh
    d = gru_cell.hidden_size

    w_ir, w_iz, w_in = w_ih[:d], w_ih[d:2*d], w_ih[2*d:]
    w_hr, w_hz, w_hn = w_hh[:d], w_hh[d:2*d], w_hh[2*d:]
    b_ir, b_iz, b_in = b_ih[:d], b_ih[d:2*d], b_ih[2*d:]
    b_hr, b_hz, b_hn = b_hh[:d], b_hh[d:2*d], b_hh[2*d:]

    r_t = torch.sigmoid(torch.nn.functional.linear(x, w_ir, b_ir) + torch.nn.functional.linear(h_prev, w_hr, b_hr))
    z_t = torch.sigmoid(torch.nn.functional.linear(x, w_iz, b_iz) + torch.nn.functional.linear(h_prev, w_hz, b_hz))
    n_t = torch.tanh(torch.nn.functional.linear(x, w_in, b_in) + r_t * torch.nn.functional.linear(h_prev, w_hn, b_hn))
    
    h_next_manual = (1 - z_t) * n_t + z_t * h_prev
    h_next_torch = gru_cell(x, h_prev)
    assert torch.allclose(h_next_manual, h_next_torch, atol=1e-5), "Manual GRU implementation mismatch!"
    
    return r_t, z_t, n_t

def main():
    out_dir = Path("results/EXP025C")
    out_dir.mkdir(parents=True, exist_ok=True)
    chk_dir = out_dir / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/EXP021_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = get_device() if config['device'] == 'auto' else config['device']
    logger.info(f"Using device: {device}")

    df = load_frozen_exp014_dataset()
    vocab = build_character_vocab(df, vocab_path="results/EXP025C/character_vocab.json")

    APPROVED_STATE_FREE_FEATURES = [
        "candidate_in_quote_chain", "candidate_is_attributed_speaker", "candidate_is_explicit_mention",
        "candidate_is_recent_mention", "chain_recency", "conversation_length", "conversation_turn_index",
        "discourse_context_length", "discourse_dialogue_position", "lexical_has_exclamation",
        "lexical_has_question_mark", "lexical_quote_length_chars", "lexical_quote_length_tokens",
        "nearest_coref_dist", "recent_mention_count",
    ]
    input_dim = len(APPROVED_STATE_FREE_FEATURES)

    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()

    scaler = StandardScaler()
    train_df[APPROVED_STATE_FREE_FEATURES] = scaler.fit_transform(train_df[APPROVED_STATE_FREE_FEATURES])
    test_df[APPROVED_STATE_FREE_FEATURES] = scaler.transform(test_df[APPROVED_STATE_FREE_FEATURES])

    train_seq = TensorSequenceDataset(train_df, APPROVED_STATE_FREE_FEATURES, feature_mode='state_free', vocab=vocab)
    test_seq = TensorSequenceDataset(test_df, APPROVED_STATE_FREE_FEATURES, feature_mode='state_free', vocab=vocab)

    def collate_fn(batch): return batch[0]
    train_loader = DataLoader(train_seq, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_seq, batch_size=1, shuffle=False, collate_fn=collate_fn)

    SEEDS = [1, 2, 3, 4, 5]
    epochs = config['epochs']
    
    # Check EXP025 accuracies for validation
    original_metrics_df = pd.read_csv("results/EXP025/seed_metrics.csv")
    gru_orig_metrics = original_metrics_df[original_metrics_df['Model'] == 'gru_normal'].set_index('Seed')['Accuracy'].to_dict()

    # Collectors for diagnostics
    diag1_records = []
    diag2_records = []
    diag3_records = []
    diag5_records = []
    diag4_results = []

    for seed in SEEDS:
        logger.info(f"========== SEED {seed} ==========")
        set_seed(seed)
        
        gru = RelationalSpeakerGRU(feature_dim=input_dim, hidden_dim=64).to(device)
        chk_path = chk_dir / f"gru_seed{seed}.pt"

        if chk_path.exists():
            logger.info(f"Loading checkpoint {chk_path}...")
            gru.load_state_dict(torch.load(chk_path, map_location=device))
        else:
            logger.info(f"Retraining model for seed {seed} (Original checkpoint unavailable)...")
            opt_gru = torch.optim.Adam(gru.parameters(), lr=config['learning_rate'])
            crit = nn.CrossEntropyLoss()
            for epoch in range(epochs):
                gru.train()
                for batch in train_loader:
                    features, mask, gold_index, _, _, _ = batch_to_device(batch, device)
                    opt_gru.zero_grad()
                    scores, _ = gru(features, mask, gold_index_for_update=gold_index)
                    loss = crit(scores.squeeze(0), gold_index.squeeze(0) if gold_index.dim() == 2 else gold_index)
                    loss.backward()
                    opt_gru.step()
            torch.save(gru.state_dict(), chk_path)

        # VALIDATION: Retrained accuracy against original EXP025
        gru.eval()
        correct_count = 0
        total_count = 0
        
        # We need normal predictions for Diagnostic 2
        normal_preds = {}
        
        with torch.no_grad():
            for batch in test_loader:
                features, mask, gold_index, q_ids, _, _ = batch_to_device(batch, device)
                scores, _ = gru(features, mask, gold_index_for_update=None)
                scores = scores.squeeze(0)
                golds = gold_index.squeeze(0) if gold_index.dim() == 2 else gold_index
                preds = torch.argmax(scores, dim=-1)
                
                probs = torch.softmax(scores, dim=-1)
                
                for i in range(len(q_ids)):
                    q_id = q_ids[i]
                    is_correct = (preds[i] == golds[i]).item()
                    correct_count += is_correct
                    total_count += 1
                    
                    normal_preds[q_id] = {
                        'pred_idx': preds[i].item(),
                        'gold_idx': golds[i].item(),
                        'gold_prob': probs[i, golds[i]].item(),
                        'max_prob': probs[i, preds[i]].item(),
                        'is_correct': is_correct
                    }
                    
        acc = correct_count / total_count
        orig_acc = gru_orig_metrics[seed]
        logger.info(f"Seed {seed} Validation: Retrained Acc={acc:.4f}, Original EXP025 Acc={orig_acc:.4f}")
        if abs(acc - orig_acc) > 0.01:
            logger.error(f"Mismatch exceeds 1.0%! Retrained: {acc:.4f}, Original: {orig_acc:.4f}")
            logger.error("Stopping script.")
            return

        # DIAGNOSTIC 1 & 5: Hidden-State Movement & Gate Statistics
        logger.info(f"Running Diagnostic 1 & 5 on Seed {seed}...")
        with torch.no_grad():
            for batch in test_loader:
                features, mask, gold_index, q_ids, _, _ = batch_to_device(batch, device)
                seq_len = features.shape[1]
                h = torch.zeros(1, gru.hidden_dim, device=device)
                h_prev = None
                h_prev_norm = 0
                
                for t in range(seq_len):
                    feats_t = features[0, t]
                    mask_t = mask[0, t]
                    
                    h_norm = torch.norm(h, p=2).item()
                    h_mean = torch.mean(h).item()
                    h_std = torch.std(h, unbiased=False).item()
                    h_var = torch.var(h, unbiased=False).item()
                    
                    if t == 0:
                        cos_sim = np.nan
                    else:
                        if h_prev_norm == 0 or h_norm == 0:
                            cos_sim = 1.0 if h_prev_norm == h_norm else 0.0
                        else:
                            cos_sim = torch.nn.functional.cosine_similarity(h, h_prev, dim=-1).item()
                            
                    h_prev = h.clone()
                    h_prev_norm = h_norm
                    
                    cand_vecs = gru.candidate_encoder(feats_t)
                    h_expanded = h.expand(cand_vecs.size(0), -1)
                    sim = torch.nn.functional.cosine_similarity(cand_vecs, h_expanded, dim=-1).unsqueeze(-1)
                    x = torch.cat([cand_vecs, h_expanded, sim], dim=-1)
                    scores_t = gru.scorer(x).squeeze(-1)
                    scores_t = scores_t.masked_fill(~mask_t, float('-inf'))
                    
                    pred_idx = torch.argmax(scores_t).item()
                    gold_idx = gold_index[t].item()
                    is_correct = int(pred_idx == gold_idx)
                    probs_t = torch.softmax(scores_t, dim=-1)
                    confidence = probs_t[pred_idx].item()
                    
                    q_id = q_ids[t]
                    
                    diag1_records.append({
                        'seed': seed,
                        'quote_id': q_id,
                        'hidden_norm': h_norm,
                        'hidden_mean': h_mean,
                        'hidden_std': h_std,
                        'hidden_variance': h_var,
                        'cos_h_t_h_prev': cos_sim,
                        'prediction_correct': is_correct,
                        'prediction_confidence': confidence
                    })
                    
                    spk_vec = cand_vecs[pred_idx].unsqueeze(0)
                    r_t, z_t, n_t = manual_gru_cell_gates(spk_vec, h, gru.gru_cell)
                    h = gru.gru_cell(spk_vec, h)
                    
                    diag5_records.append({
                        'seed': seed,
                        'quote_id': q_id,
                        'update_gate_mean': z_t.mean().item(),
                        'update_gate_std': z_t.std(unbiased=False).item(),
                        'reset_gate_mean': r_t.mean().item(),
                        'reset_gate_std': r_t.std(unbiased=False).item(),
                        'update_gate_lt_0_1': (z_t < 0.1).float().mean().item(),
                        'update_gate_gt_0_9': (z_t > 0.9).float().mean().item()
                    })
                    
        # DIAGNOSTIC 2: Prediction Equivalence
        logger.info(f"Running Diagnostic 2 on Seed {seed}...")
        modes = [
            ("reset_hidden_each_quote", {"ablate_memory": True}),
            ("zero_update", {"ablate_memory": True}),
            ("shuffled_update", {"ablate_shuffle": True}),
            ("teacher_forced_eval_diagnostic", {}) # For TF, we need gold_index
        ]
        
        for mode, kwargs in modes:
            top1_agreements = 0
            normal_corr_ablation_wrong = 0
            ablation_corr_normal_wrong = 0
            both_correct = 0
            both_wrong = 0
            sum_abs_gold_prob_diff = 0.0
            sum_abs_max_prob_diff = 0.0
            total = 0
            
            torch.manual_seed(42) # Ensure shuffled_update reproducibility
            with torch.no_grad():
                for batch in test_loader:
                    features, mask, gold_index, q_ids, _, _ = batch_to_device(batch, device)
                    
                    if mode == "teacher_forced_eval_diagnostic":
                        scores, _ = gru(features, mask, gold_index_for_update=gold_index, **kwargs)
                    else:
                        scores, _ = gru(features, mask, gold_index_for_update=None, **kwargs)
                        
                    scores = scores.squeeze(0)
                    golds = gold_index.squeeze(0) if gold_index.dim() == 2 else gold_index
                    preds = torch.argmax(scores, dim=-1)
                    probs = torch.softmax(scores, dim=-1)
                    
                    for i in range(len(q_ids)):
                        q_id = q_ids[i]
                        norm = normal_preds[q_id]
                        
                        a_pred = preds[i].item()
                        a_gold = golds[i].item()
                        a_gold_prob = probs[i, a_gold].item()
                        a_max_prob = probs[i, a_pred].item()
                        a_correct = (a_pred == a_gold)
                        
                        n_pred = norm['pred_idx']
                        n_correct = norm['is_correct']
                        n_gold_prob = norm['gold_prob']
                        n_max_prob = norm['max_prob']
                        
                        if n_pred == a_pred: top1_agreements += 1
                        if n_correct and not a_correct: normal_corr_ablation_wrong += 1
                        if not n_correct and a_correct: ablation_corr_normal_wrong += 1
                        if n_correct and a_correct: both_correct += 1
                        if not n_correct and not a_correct: both_wrong += 1
                        
                        sum_abs_gold_prob_diff += abs(n_gold_prob - a_gold_prob)
                        sum_abs_max_prob_diff += abs(n_max_prob - a_max_prob)
                        total += 1
                        
            diag2_records.append({
                'seed': seed,
                'comparison_mode': mode,
                'top1_agreement_percent': top1_agreements / total,
                'normal_correct_ablation_wrong': normal_corr_ablation_wrong,
                'ablation_correct_normal_wrong': ablation_corr_normal_wrong,
                'both_correct': both_correct,
                'both_wrong': both_wrong,
                'mean_abs_gold_prob_difference': sum_abs_gold_prob_diff / total,
                'mean_abs_max_prob_difference': sum_abs_max_prob_diff / total
            })

        # DIAGNOSTIC 3: Scorer Hidden-State Weight Usage
        w = gru.scorer[0].weight.data
        d = gru.hidden_dim
        w_cand = w[:, :d]
        w_hidden = w[:, d:2*d]
        norm_cand = torch.norm(w_cand, p='fro').item()
        norm_hidden = torch.norm(w_hidden, p='fro').item()
        
        diag3_records.append({
            'seed': seed,
            'candidate_block_weight_norm': norm_cand,
            'hidden_block_weight_norm': norm_hidden,
            'hidden_to_candidate_norm_ratio': norm_hidden / norm_cand if norm_cand > 0 else 0
        })

        # DIAGNOSTIC 4: Previous-Speaker Probe
        logger.info(f"Running Diagnostic 4 on Seed {seed}...")
        def collect_probe_dataset(loader, is_train=True):
            X_h, X_c, X_hc, Y = [], [], [], []
            with torch.no_grad():
                for batch in loader:
                    features, mask, gold_index, q_ids, cand_ids, gold_spk_ids = batch_to_device(batch, device)
                    seq_len = features.shape[1]
                    h = torch.zeros(1, gru.hidden_dim, device=device)
                    
                    for t in range(seq_len):
                        feats_t = features[0, t]
                        mask_t = mask[0, t]
                        cand_vecs = gru.candidate_encoder(feats_t)
                        
                        for c in range(mask_t.size(0)):
                            if not mask_t[c]: continue
                            
                            is_prev_speaker = False
                            if t > 0:
                                prev_gold_spk_id = gold_spk_ids[t-1].item()
                                if prev_gold_spk_id > 0 and cand_ids[t, c].item() == prev_gold_spk_id:
                                    is_prev_speaker = True
                                    
                            X_h.append(h.clone().squeeze(0))
                            X_c.append(cand_vecs[c].clone())
                            X_hc.append(torch.cat([h.clone().squeeze(0), cand_vecs[c].clone()]))
                            Y.append(torch.tensor([1.0 if is_prev_speaker else 0.0], device=device))
                            
                        # Update GRU (use gold_index if train, else predicted)
                        x = torch.cat([cand_vecs, h.expand(cand_vecs.size(0), -1), 
                                     torch.nn.functional.cosine_similarity(cand_vecs, h.expand(cand_vecs.size(0), -1), dim=-1).unsqueeze(-1)], dim=-1)
                        scores_t = gru.scorer(x).squeeze(-1).masked_fill(~mask_t, float('-inf'))
                        
                        spk_idx = gold_index[t].item() if is_train else torch.argmax(scores_t).item()
                        spk_vec = cand_vecs[spk_idx].unsqueeze(0)
                        h = gru.gru_cell(spk_vec, h)
                        
            return torch.stack(X_h), torch.stack(X_c), torch.stack(X_hc), torch.stack(Y)

        X_h_train, X_c_train, X_hc_train, Y_train = collect_probe_dataset(train_loader, is_train=True)
        X_h_test, X_c_test, X_hc_test, Y_test = collect_probe_dataset(test_loader, is_train=False)

        def train_and_eval_probe(X_train, Y_train, X_test, Y_test, input_dim):
            probe = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
            opt = torch.optim.Adam(probe.parameters(), lr=0.01)
            crit = nn.BCEWithLogitsLoss()
            
            dataset = TensorDataset(X_train, Y_train)
            loader = DataLoader(dataset, batch_size=256, shuffle=True)
            
            for ep in range(10):
                probe.train()
                for bx, by in loader:
                    opt.zero_grad()
                    loss = crit(probe(bx), by)
                    loss.backward()
                    opt.step()
                    
            probe.eval()
            with torch.no_grad():
                preds = (torch.sigmoid(probe(X_test)) > 0.5).float()
                acc = (preds == Y_test).float().mean().item()
            return acc

        acc_h = train_and_eval_probe(X_h_train, Y_train, X_h_test, Y_test, 64)
        acc_c = train_and_eval_probe(X_c_train, Y_train, X_c_test, Y_test, 64)
        acc_hc = train_and_eval_probe(X_hc_train, Y_train, X_hc_test, Y_test, 128)
        acc_false = (torch.zeros_like(Y_test) == Y_test).float().mean().item()
        
        diag4_results.append({
            'seed': seed,
            'baseline_always_false': acc_false,
            'probe_hidden_only': acc_h,
            'probe_candidate_only': acc_c,
            'probe_hidden_and_candidate': acc_hc
        })

    # Save DataFrames
    pd.DataFrame(diag1_records).to_csv(out_dir / "hidden_state_movement_by_seed.csv", index=False)
    
    d1_df = pd.DataFrame(diag1_records)
    d1_summary = d1_df.groupby('seed').agg(
        mean_hidden_norm=('hidden_norm', 'mean'),
        std_hidden_norm=('hidden_norm', 'std'),
        mean_hidden_variance=('hidden_variance', 'mean'),
        mean_cos_h_t_h_prev=('cos_h_t_h_prev', lambda x: x.dropna().mean()),
        median_cos_h_t_h_prev=('cos_h_t_h_prev', lambda x: x.dropna().median()),
        percent_cos_gt_0_99=('cos_h_t_h_prev', lambda x: (x.dropna() > 0.99).mean()),
        percent_cos_gt_0_95=('cos_h_t_h_prev', lambda x: (x.dropna() > 0.95).mean())
    ).reset_index()
    d1_summary.to_csv(out_dir / "hidden_state_movement_summary.csv", index=False)
    
    pd.DataFrame(diag2_records).to_csv(out_dir / "prediction_equivalence_by_seed.csv", index=False)
    pd.DataFrame(diag3_records).to_csv(out_dir / "scorer_hidden_weight_usage.csv", index=False)
    pd.DataFrame(diag4_results).to_csv(out_dir / "previous_speaker_probe.csv", index=False)
    
    d5_df = pd.DataFrame(diag5_records)
    d5_summary = d5_df.groupby('seed').mean().reset_index().drop(columns=['quote_id'])
    d5_summary.to_csv(out_dir / "gru_gate_statistics.csv", index=False)
    
    # Generate Final Report
    report = []
    report.append("# EXP025C GRU Memory Utilization Audit\n")
    report.append("> EXP025C retrains the EXP025 GRU configuration because original checkpoints were unavailable. Therefore, EXP025C diagnoses the same architecture and training protocol, not the exact original trained parameter instances.\n")
    report.append("## Diagnostic 1: Hidden-State Movement")
    report.append(d1_summary.to_markdown(index=False) + "\n")
    
    report.append("## Diagnostic 2: Prediction Equivalence")
    report.append("> This explains the behavior of the EXP025 architecture under the same training protocol.\n")
    report.append(pd.DataFrame(diag2_records).groupby('comparison_mode')[['top1_agreement_percent', 'both_correct']].mean().reset_index().to_markdown(index=False) + "\n")
    
    report.append("## Diagnostic 3: Scorer Hidden-State Weight Usage")
    report.append(pd.DataFrame(diag3_records).to_markdown(index=False) + "\n")
    
    report.append("## Diagnostic 4: Previous-Speaker Probe")
    report.append(pd.DataFrame(diag4_results).to_markdown(index=False) + "\n")
    
    report.append("## Diagnostic 5: Gate Statistics")
    report.append(d5_summary.to_markdown(index=False) + "\n")
    
    report.append("## Conclusions\n")
    
    # Analyze conclusions
    d1_mean_cos = d1_summary['mean_cos_h_t_h_prev'].mean()
    d2_top1_reset = pd.DataFrame(diag2_records).query("comparison_mode == 'reset_hidden_each_quote'")['top1_agreement_percent'].mean()
    d3_ratio = pd.DataFrame(diag3_records)['hidden_to_candidate_norm_ratio'].mean()
    d4 = pd.DataFrame(diag4_results).mean()
    
    report.append("EXP025C does not attempt to improve the GRU.")
    report.append("It only explains why the EXP025 GRU failed to provide robust memory.\n")
    
    if d1_mean_cos > 0.99:
        report.append("- **Conclusion**: The hidden state does not move meaningfully over time. It collapsed to a static representation.")
    elif d2_top1_reset > 0.99 or d3_ratio < 0.1:
        report.append("- **Conclusion**: The hidden state moves, but the scorer largely ignores it.")
    elif d4['probe_hidden_and_candidate'] <= d4['probe_candidate_only'] + 0.01:
        report.append("- **Conclusion**: The hidden state moves and is used, but it does not encode useful speaker-history information (hidden+candidate probe performs no better than candidate-only).")
    else:
        report.append("- **Conclusion**: The hidden state encodes speaker-history information, but the scorer fails to integrate it effectively for attribution.")

    with open(out_dir / "EXP025C_MEMORY_UTILIZATION_REPORT.md", "w") as f:
        f.write("\n".join(report))
        
    logger.info("EXP025C execution complete.")

if __name__ == "__main__":
    main()
