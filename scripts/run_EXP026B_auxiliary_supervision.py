import os
import sys
import json
import yaml
import torch
import logging
import random
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.metrics import average_precision_score

# Add root directory to path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.device import get_device
from src.evaluation.runner import load_frozen_exp014_dataset
from src.neural.sequence_dataset import build_character_vocab, TensorSequenceDataset
from src.neural.models import (
    EXP026ACandidateOnlyScorer,
    EXP026ABilinearSpeakerGRU,
    EXP026AParameterMatchedNoMemoryScorer,
    EXP026BBilinearSpeakerGRUWithAuxiliary,
    compute_masked_previous_speaker_ce
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants to align with EXP026A
APPROVED_STATE_FREE_FEATURES = [
    "candidate_in_quote_chain", "candidate_is_attributed_speaker", "candidate_is_explicit_mention",
    "candidate_is_recent_mention", "chain_recency", "conversation_length", "conversation_turn_index",
    "discourse_context_length", "discourse_dialogue_position", "lexical_has_exclamation",
    "lexical_has_question_mark", "lexical_quote_length_chars", "lexical_quote_length_tokens",
    "nearest_coref_dist", "recent_mention_count"
]

FORBIDDEN_FEATURES = {
    "candidate_is_last_speaker", "candidate_is_previous_speaker", "candidate_in_participant_stack",
    "candidate_stack_depth", "conversation_speaker_change", "conv_active_id", "conv_interruption_distance"
}

STOPPING_RULE = "Regardless of statistical significance, EXP026B is the final GRU-state learning test."

VAL_NOVELS = [
    "TheSignOfTheFour", "TheSportOfTheGods", "TheSunAlsoRises",
    "WhereAngelsFearToTread", "WinnieThePooh"
]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_one(batch):
    return batch[0]

def batch_to_device(batch, device):
    features = batch.candidate_features
    cids = batch.candidate_ids
    mask = batch.candidate_mask
    if features.dim() == 3:
        features = features.unsqueeze(0)
    if cids.dim() == 2:
        cids = cids.unsqueeze(0)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    return (
        features.to(device),
        cids.to(device),
        mask.to(device),
        batch.gold_index.to(device),
        batch.quote_ids,
        batch.novel_id,
        batch.gold_speaker_id.to(device)
    )

def reverse_vocab(vocab):
    rev = {}
    for k, v in vocab.items():
        # k is novel::character
        parts = k.split("::")
        char_name = parts[1] if len(parts) > 1 else parts[0]
        rev[v] = char_name
    return rev

def display_candidate(novel_id, cid, rev_vocab):
    return rev_vocab.get(cid, f"Unknown_{cid}")

def load_quote_types(df):
    q_info_dir = Path("data/raw/pdnc/data")
    type_mappings = {}
    for novel in df['novel'].unique():
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
                type_mappings[q_id] = row.get("quoteType")
    return type_mappings

def make_parameter_matched_control(input_dim, target_params, base_hidden_dim=64):
    current_dim = base_hidden_dim
    while True:
        candidate_encoder = nn.Sequential(
            nn.Linear(input_dim, current_dim),
            nn.ReLU(),
            nn.Linear(current_dim, current_dim)
        )
        candidate_score_branch = nn.Sequential(
            nn.Linear(current_dim, current_dim // 2),
            nn.ReLU(),
            nn.Linear(current_dim // 2, 1)
        )
        total = sum(p.numel() for p in candidate_encoder.parameters()) + sum(p.numel() for p in candidate_score_branch.parameters())
        if total >= target_params:
            break
        current_dim += 2
    return EXP026AParameterMatchedNoMemoryScorer(input_dim, current_dim)

def run_preflight_audit(train_seq, test_seq, type_mappings, out_dir):
    logger.info("Running preflight auxiliary target audit...")
    audit = {
        "number_of_quotes": 0,
        "quotes_with_previous_speaker": 0,
        "quotes_where_previous_speaker_in_candidate_set": 0,
        "auxiliary_supervision_coverage": 0.0,
        "mean_candidate_set_size": 0.0,
        "positive_auxiliary_labels": 0,
        "negative_auxiliary_labels": 0,
        "positive_negative_ratio": 0.0,
        "coverage_by_quote_type": {},
        "coverage_by_novel": {}
    }

    type_counts = defaultdict(lambda: {"total": 0, "covered": 0})
    novel_counts = defaultdict(lambda: {"total": 0, "covered": 0})
    candidate_sizes = []

    for seq in list(train_seq.sequences) + list(test_seq.sequences):
        novel = seq.novel_id
        gold_spk_ids = seq.gold_speaker_id
        candidate_ids = seq.candidate_ids
        mask = seq.candidate_mask
        quote_ids = seq.quote_ids

        for t in range(len(quote_ids)):
            q_id = quote_ids[t]
            q_type = type_mappings.get(q_id, "Unknown")

            audit["number_of_quotes"] += 1
            type_counts[q_type]["total"] += 1
            novel_counts[novel]["total"] += 1

            cands_t = candidate_ids[t]
            mask_t = mask[t]
            num_cands = mask_t.sum().item()
            candidate_sizes.append(num_cands)

            if t > 0:
                prev_spk = gold_spk_ids[t-1].item()
                if prev_spk > 0:
                    audit["quotes_with_previous_speaker"] += 1
                    
                    # Check if present and valid in candidate set
                    idx_matches = (cands_t == prev_spk) & mask_t
                    is_covered = idx_matches.any().item()

                    if is_covered:
                        audit["quotes_where_previous_speaker_in_candidate_set"] += 1
                        type_counts[q_type]["covered"] += 1
                        novel_counts[novel]["covered"] += 1
                        
                        audit["positive_auxiliary_labels"] += 1
                        audit["negative_auxiliary_labels"] += (num_cands - 1)
                    else:
                        audit["negative_auxiliary_labels"] += num_cands

    if audit["number_of_quotes"] > 0:
        audit["auxiliary_supervision_coverage"] = audit["quotes_where_previous_speaker_in_candidate_set"] / audit["number_of_quotes"]
        audit["mean_candidate_set_size"] = float(np.mean(candidate_sizes))
    if audit["negative_auxiliary_labels"] > 0:
        audit["positive_negative_ratio"] = audit["positive_auxiliary_labels"] / audit["negative_auxiliary_labels"]

    for k, v in type_counts.items():
        audit["coverage_by_quote_type"][k] = {
            "total": v["total"],
            "covered": v["covered"],
            "coverage": v["covered"] / v["total"] if v["total"] > 0 else 0.0
        }
    for k, v in novel_counts.items():
        audit["coverage_by_novel"][k] = {
            "total": v["total"],
            "covered": v["covered"],
            "coverage": v["covered"] / v["total"] if v["total"] > 0 else 0.0
        }

    with open(out_dir / "auxiliary_target_audit.json", "w") as f:
        json.dump(audit, f, indent=4)
    logger.info("Preflight audit saved to results/EXP026B/auxiliary_target_audit.json")
    return audit

def train_model(model, train_loader, device, lr, epochs, variant, lambda_val=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    attr_criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_attr_loss = 0.0
        total_aux_loss = 0.0
        for batch in train_loader:
            features, cids, mask, gold_index, _, _, gold_spk = batch_to_device(batch, device)
            optimizer.zero_grad()

            if variant.startswith("nomemory"):
                scores, _ = model(features, mask)
                loss = attr_criterion(scores.squeeze(0), gold_index.squeeze(0) if gold_index.dim() == 2 else gold_index)
            else:
                if isinstance(model, EXP026BBilinearSpeakerGRUWithAuxiliary):
                    scores, _, aux_scores = model(features, mask, gold_index_for_update=gold_index)
                    loss_attr = attr_criterion(scores.squeeze(0), gold_index.squeeze(0) if gold_index.dim() == 2 else gold_index)
                    loss_aux = compute_masked_previous_speaker_ce(aux_scores, cids, gold_spk, mask)
                    loss = loss_attr + lambda_val * loss_aux
                    total_aux_loss += loss_aux.item()
                else:
                    scores, _ = model(features, mask, gold_index_for_update=gold_index)
                    loss = attr_criterion(scores.squeeze(0), gold_index.squeeze(0) if gold_index.dim() == 2 else gold_index)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # logger.info("%s epoch %d/%d loss %.4f", variant, epoch + 1, epochs, total_loss / len(train_loader))

def evaluate_model_on_split(model, dataloader, device, variant, lambda_val=0.0):
    """Evaluates the model on validation or test set and returns loss, metrics, and aux task outputs."""
    model.eval()
    attr_criterion = nn.CrossEntropyLoss(reduction='sum')
    
    total_attr_loss = 0.0
    total_aux_loss = 0.0
    correct_attr = 0
    total_attr = 0
    
    correct_aux = 0
    total_aux_steps = 0
    aux_mrr_sum = 0.0
    
    aux_scores_all = []
    aux_targets_all = []

    with torch.no_grad():
        for batch in dataloader:
            features, cids, mask, gold_index, _, _, gold_spk = batch_to_device(batch, device)
            
            if isinstance(model, EXP026BBilinearSpeakerGRUWithAuxiliary):
                scores, _, aux_scores = model(features, mask, gold_index_for_update=None)
                loss_aux = compute_masked_previous_speaker_ce(aux_scores, cids, gold_spk, mask)
                total_aux_loss += loss_aux.item() * features.size(1)
                
                # Compute auxiliary metrics
                aux_scores_sq = aux_scores.squeeze(0) # [seq_len, max_cand]
                cids_sq = cids.squeeze(0)
                gold_spk_sq = gold_spk.squeeze(0)
                mask_sq = mask.squeeze(0)
                
                for t in range(features.size(1)):
                    if t == 0: continue
                    prev_spk = gold_spk_sq[t-1].item()
                    if prev_spk <= 0: continue
                    
                    idx_matches = (cids_sq[t] == prev_spk) & mask_sq[t]
                    matching_indices = torch.nonzero(idx_matches, as_tuple=False).flatten()
                    
                    if len(matching_indices) == 1:
                        target_idx = matching_indices[0].item()
                        pred_aux_idx = torch.argmax(aux_scores_sq[t]).item()
                        
                        # Accuracy
                        if pred_aux_idx == target_idx:
                            correct_aux += 1
                        
                        # MRR
                        sorted_aux = torch.argsort(aux_scores_sq[t], descending=True)
                        ranks = (sorted_aux == target_idx).nonzero(as_tuple=True)[0]
                        if len(ranks) > 0:
                            aux_mrr_sum += 1.0 / (ranks[0].item() + 1)
                        total_aux_steps += 1
                        
                        # Save for PR-AUC
                        probs_t = torch.softmax(aux_scores_sq[t], dim=-1)
                        for c in range(mask_sq.size(1)):
                            if mask_sq[t, c]:
                                aux_scores_all.append(probs_t[c].item())
                                aux_targets_all.append(1.0 if c == target_idx else 0.0)
            else:
                scores, _ = model(features, mask, gold_index_for_update=None)
            
            scores_sq = scores.squeeze(0)
            golds_sq = gold_index.squeeze(0) if gold_index.dim() == 2 else gold_index
            loss_attr = attr_criterion(scores_sq, golds_sq)
            total_attr_loss += loss_attr.item()
            
            preds = torch.argmax(scores_sq, dim=-1)
            correct_attr += (preds == golds_sq).sum().item()
            total_attr += golds_sq.size(0)

    acc = correct_attr / total_attr
    mean_attr_loss = total_attr_loss / total_attr
    mean_aux_loss = total_aux_loss / total_attr if isinstance(model, EXP026BBilinearSpeakerGRUWithAuxiliary) else 0.0
    aux_acc = correct_aux / total_aux_steps if total_aux_steps > 0 else 0.0
    aux_mrr = aux_mrr_sum / total_aux_steps if total_aux_steps > 0 else 0.0
    
    aux_pr_auc = 0.0
    if aux_scores_all and sum(aux_targets_all) > 0:
        aux_pr_auc = average_precision_score(aux_targets_all, aux_scores_all)

    return {
        "Accuracy": acc,
        "Attribution_Loss": mean_attr_loss,
        "Auxiliary_Loss": mean_aux_loss,
        "Auxiliary_Accuracy": aux_acc,
        "Auxiliary_MRR": aux_mrr,
        "Auxiliary_PR_AUC": aux_pr_auc
    }

def run_model_scores(model, features, mask, gold_index, variant, mode):
    if variant.startswith("nomemory"):
        res = model(features, mask)
        if isinstance(res, tuple):
            return res[0], res[1]
        return res, None

    if gold_index.dim() == 1:
        gold_index = gold_index.unsqueeze(0)

    if mode == "teacher_forced_eval_diagnostic":
        res = model(features, mask, gold_index_for_update=gold_index)
    elif mode == "zero_state":
        res = model(features, mask, gold_index_for_update=None, ablate_memory=True)
    elif mode == "shuffled_update":
        torch.manual_seed(42)
        res = model(features, mask, gold_index_for_update=None, ablate_shuffle=True)
    else:
        res = model(features, mask, gold_index_for_update=None)
        
    return res[0], res[1] # always return attribution scores and interactions

def evaluate_scorer(model, dataloader, device, type_mappings, vocab, variant, mode):
    model.eval()
    rows = []
    rev_vocab = reverse_vocab(vocab)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for batch in dataloader:
            features, cids, mask, gold_index, q_ids, novel_id, _ = batch_to_device(batch, device)
            scores, diagnostics = run_model_scores(model, features, mask, gold_index, variant, mode)
            scores = scores.squeeze(0)
            mask_2d = mask.squeeze(0)
            cids_2d = cids.squeeze(0)
            gold_index_1d = gold_index.squeeze(0) if gold_index.dim() == 2 else gold_index
            losses = loss_fn(scores, gold_index_1d).detach().cpu().numpy()
            probs = torch.softmax(scores, dim=-1)
            sorted_indices = torch.argsort(scores, dim=-1, descending=True)

            interaction_diag = None
            if diagnostics is not None and "bilinear" in variant:
                interaction_diag = diagnostics.squeeze(0)

            for i, q_id in enumerate(q_ids):
                valid = mask_2d[i].detach().cpu().numpy().astype(bool)
                gold = gold_index_1d[i].item()
                pred = sorted_indices[i, 0].item()
                ranks = (sorted_indices[i] == gold).nonzero(as_tuple=True)[0]
                rank = ranks[0].item() + 1 if len(ranks) > 0 else 999
                valid_scores = scores[i, mask_2d[i]].detach().cpu().tolist()
                valid_probs = probs[i, mask_2d[i]].detach().cpu().tolist()
                valid_ids = cids_2d[i, mask_2d[i]].detach().cpu().tolist()
                interaction_values = []
                if interaction_diag is not None:
                    interaction_values = interaction_diag[i, mask_2d[i]].detach().cpu().tolist()

                rows.append({
                    "seed": None,
                    "novel": novel_id,
                    "quote_id": q_id,
                    "quote_type": type_mappings.get(q_id, "Unknown"),
                    "gold_candidate": display_candidate(novel_id, cids_2d[i, gold].item(), rev_vocab),
                    "gold_candidate_index": gold,
                    "predicted_candidate": display_candidate(novel_id, cids_2d[i, pred].item(), rev_vocab),
                    "predicted_candidate_index": pred,
                    "candidate_ids": json.dumps(valid_ids),
                    "candidate_scores": json.dumps(valid_scores),
                    "candidate_probabilities": json.dumps(valid_probs),
                    "correct": bool(pred == gold),
                    "pred_rank": rank,
                    "loss": float(losses[i]),
                    "gold_probability": float(probs[i, gold].item()),
                    "evaluation_mode": mode,
                    "interaction_scores": json.dumps(interaction_values) if interaction_values else json.dumps([0.0]*len(valid_scores))
                })
    return pd.DataFrame(rows)

def compute_overall_metrics(preds_df):
    acc = preds_df["correct"].mean()
    implicit = preds_df[preds_df["quote_type"] == "Implicit"]["correct"].mean()
    anaphoric = preds_df[preds_df["quote_type"] == "Anaphoric"]["correct"].mean()
    mrr = (1.0 / preds_df["pred_rank"]).mean()
    log_loss = preds_df["loss"].mean()
    return {
        "Accuracy": acc,
        "Implicit_Accuracy": implicit,
        "Anaphoric_Accuracy": anaphoric,
        "MRR": mrr,
        "LogLoss": log_loss
    }

def run_probe_controls(model, test_seq, test_df_scaled, device, seed):
    """Evaluates the training-time success of previous speaker encoding on the test set."""
    X_h_ar, X_c_ar, X_hc_ar, Y_ar = [], [], [], []
    X_h_shuf, X_hc_shuf = [], []
    
    model.eval()
    with torch.no_grad():
        for seq in test_seq.sequences:
            device_seq = seq.candidate_features.to(device)
            mask_seq = seq.candidate_mask.to(device)
            cids = seq.candidate_ids.to(device)
            gold_spk = seq.gold_speaker_id.to(device)
            
            seq_len, max_cand, _ = device_seq.shape
            
            # Autoregressive states
            h_ar = torch.zeros(1, model.hidden_dim, device=device)
            # Shuffled states
            h_shuf = torch.zeros(1, model.hidden_dim, device=device)
            
            for t in range(seq_len):
                feats_t = device_seq[t]
                mask_t = mask_seq[t]
                cand_vecs = model.encode_candidates(feats_t)
                
                # Check previous speaker target
                is_prev_speaker = False
                prev_spk_id = gold_spk[t-1].item() if t > 0 else 0
                
                for c in range(mask_t.size(0)):
                    if not mask_t[c]: continue
                    
                    is_prev = (prev_spk_id > 0 and cids[t, c].item() == prev_spk_id)
                    
                    X_h_ar.append(h_ar.clone().squeeze(0))
                    X_c_ar.append(cand_vecs[c].clone())
                    X_hc_ar.append(torch.cat([h_ar.clone().squeeze(0), cand_vecs[c].clone()]))
                    
                    X_h_shuf.append(h_shuf.clone().squeeze(0))
                    X_hc_shuf.append(torch.cat([h_shuf.clone().squeeze(0), cand_vecs[c].clone()]))
                    
                    Y_ar.append(1.0 if is_prev else 0.0)
                    
                # Update AR state (predicted)
                h_ar_exp = h_ar.expand(max_cand, -1)
                scores_t = model.score(cand_vecs, h_ar_exp).masked_fill(~mask_t, float('-inf'))
                pred_idx = torch.argmax(scores_t).item()
                h_ar = model.gru_cell(cand_vecs[pred_idx].unsqueeze(0), h_ar)
                
                # Update Shuffled state (shuffled valid index)
                valid_idx = torch.nonzero(mask_t, as_tuple=False).flatten()
                shuf_idx = valid_idx[torch.randint(0, len(valid_idx), (1,))].item() if len(valid_idx) > 0 else 0
                h_shuf = model.gru_cell(cand_vecs[shuf_idx].unsqueeze(0), h_shuf)

    X_h_ar = torch.stack(X_h_ar)
    X_c_ar = torch.stack(X_c_ar)
    X_hc_ar = torch.stack(X_hc_ar)
    X_h_shuf = torch.stack(X_h_shuf)
    X_hc_shuf = torch.stack(X_hc_shuf)
    Y_ar = np.array(Y_ar)
    
    # Train simple linear probes (logistic regression) over these features
    def train_and_eval_logistic_probe(X, Y):
        # We can implement a simple PyTorch logistic regression
        in_dim = X.shape[1]
        probe = nn.Linear(in_dim, 1).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.1)
        
        # Calculate balanced weights
        num_pos = Y.sum()
        num_neg = len(Y) - num_pos
        pos_weight = torch.tensor([num_neg / max(1.0, num_pos)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        dataset = TensorDataset(X.to(device), torch.tensor(Y, dtype=torch.float32, device=device).unsqueeze(-1))
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        for epoch in range(10):
            probe.train()
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(probe(bx), by)
                loss.backward()
                optimizer.step()
                
        probe.eval()
        with torch.no_grad():
            logits = probe(X.to(device))
            probs = torch.sigmoid(logits).cpu().squeeze(-1).numpy()
            
        pr_auc = average_precision_score(Y, probs)
        return pr_auc

    pr_cand = train_and_eval_logistic_probe(X_c_ar, Y_ar)
    pr_hc = train_and_eval_logistic_probe(X_hc_ar, Y_ar)
    pr_shuf = train_and_eval_logistic_probe(X_hc_shuf, Y_ar)
    
    # Zero state hidden+candidate (which is equivalent to zero hidden concatenated)
    X_hz = torch.cat([torch.zeros(X_h_ar.shape[0], X_h_ar.shape[1]), X_c_ar], dim=-1)
    pr_zero = train_and_eval_logistic_probe(X_hz, Y_ar)
    
    return [
        {"Seed": seed, "Probe": "candidate_only", "Previous_Speaker_PR_AUC": pr_cand},
        {"Seed": seed, "Probe": "candidate_plus_aligned_hidden", "Previous_Speaker_PR_AUC": pr_hc},
        {"Seed": seed, "Probe": "candidate_plus_shuffled_hidden", "Previous_Speaker_PR_AUC": pr_shuf},
        {"Seed": seed, "Probe": "candidate_plus_zero_hidden", "Previous_Speaker_PR_AUC": pr_zero}
    ]

def main():
    out_dir = Path("results/EXP026B")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/EXP021_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = get_device() if config['device'] == 'auto' else config['device']
    logger.info("Using device: %s", device)

    # 1. Load Frozen EXP014 Dataset
    df = load_frozen_exp014_dataset()
    vocab = build_character_vocab(df, vocab_path=str(out_dir / "character_vocab.json"))

    # ASSERT EQUIVALENCE: Check features and shapes are identical to EXP026A configuration
    assert len(APPROVED_STATE_FREE_FEATURES) == 15, "Feature list dimension mismatch!"
    missing = sorted(set(APPROVED_STATE_FREE_FEATURES) - set(df.columns))
    assert not missing, f"Missing approved features: {missing}"
    
    present_forbidden = sorted(set(APPROVED_STATE_FREE_FEATURES) & FORBIDDEN_FEATURES)
    assert not present_forbidden, f"Forbidden features present in approved features: {present_forbidden}"

    # Partition training dataset into 20 Train Novels and 5 Validation Novels
    # Alphabetical last 5 training novels are used as the validation split
    train_all_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    train_novels = sorted(train_all_df['novel'].unique())
    validation_novels = VAL_NOVELS
    train_sub_novels = [n for n in train_novels if n not in validation_novels]
    
    assert len(train_novels) == 25, f"Expected 25 train novels, got {len(train_novels)}"
    assert len(validation_novels) == 5, f"Expected 5 validation novels, got {len(validation_novels)}"
    assert len(train_sub_novels) == 20, f"Expected 20 train sub novels, got {len(train_sub_novels)}"

    train_df = train_all_df[train_all_df['novel'].isin(train_sub_novels)].copy()
    val_df = train_all_df[train_all_df['novel'].isin(validation_novels)].copy()

    # Scaling
    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()
    test_df_scaled = test_df.copy()
    
    # We also need a full-train scaler for the final testing evaluation
    scaler_full = StandardScaler()
    train_all_scaled = train_all_df.copy()
    train_all_scaled[APPROVED_STATE_FREE_FEATURES] = scaler_full.fit_transform(train_all_df[APPROVED_STATE_FREE_FEATURES])

    # Fit scaling on the training sub-split
    train_df_scaled[APPROVED_STATE_FREE_FEATURES] = scaler.fit_transform(train_df[APPROVED_STATE_FREE_FEATURES])
    val_df_scaled[APPROVED_STATE_FREE_FEATURES] = scaler.transform(val_df[APPROVED_STATE_FREE_FEATURES])
    test_df_scaled[APPROVED_STATE_FREE_FEATURES] = scaler.transform(test_df[APPROVED_STATE_FREE_FEATURES])

    # Setup datasets
    train_seq = TensorSequenceDataset(train_df_scaled, APPROVED_STATE_FREE_FEATURES, feature_mode='all', vocab=vocab)
    val_seq = TensorSequenceDataset(val_df_scaled, APPROVED_STATE_FREE_FEATURES, feature_mode='all', vocab=vocab)
    test_seq = TensorSequenceDataset(test_df_scaled, APPROVED_STATE_FREE_FEATURES, feature_mode='all', vocab=vocab)
    
    # Setup full-train dataset for final testing evaluation
    train_all_seq = TensorSequenceDataset(train_all_scaled, APPROVED_STATE_FREE_FEATURES, feature_mode='all', vocab=vocab)

    train_loader = DataLoader(train_seq, batch_size=1, shuffle=True, collate_fn=collate_one)
    val_loader = DataLoader(val_seq, batch_size=1, shuffle=False, collate_fn=collate_one)
    test_loader = DataLoader(test_seq, batch_size=1, shuffle=False, collate_fn=collate_one)
    
    train_all_loader = DataLoader(train_all_seq, batch_size=1, shuffle=True, collate_fn=collate_one)

    type_mappings = load_quote_types(df)
    input_dim = len(APPROVED_STATE_FREE_FEATURES)
    
    is_cpu_test = os.environ.get("CPU_TEST_RUN") == "1"
    seeds = [1] if is_cpu_test else [1, 2, 3, 4, 5]
    epochs = 1 if is_cpu_test else config['epochs']

    # Save running configuration details
    run_config = {
        "experiment_name": "EXP026B_auxiliary_supervision",
        "stopping_rule": STOPPING_RULE,
        "seeds": seeds,
        "epochs": epochs,
        "learning_rate": config["learning_rate"],
        "input_dim": input_dim,
        "hidden_dim": 64,
        "cpu_test_run": is_cpu_test,
        "lambda_values": [0.1, 0.3, 1.0],
        "validation_novels": validation_novels
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=4)

    # 2. Preflight target audit
    preflight_audit = run_preflight_audit(train_all_seq, test_seq, type_mappings, out_dir)

    # ==================== STAGE 1: Validation Sweep ====================
    logger.info("==================== STAGE 1: VALIDATION SWEEP ====================")
    val_records = []
    
    # Lambda values to sweep
    lambdas = [0.1, 0.3, 1.0]

    for seed in seeds:
        logger.info(f"--- Sweeping lambdas for Seed {seed} ---")
        # 1. Sweep active lambdas
        for l_val in lambdas:
            set_seed(seed)
            model = EXP026BBilinearSpeakerGRUWithAuxiliary(input_dim, hidden_dim=64).to(device)
            model.feature_names = APPROVED_STATE_FREE_FEATURES
            
            train_model(model, train_loader, device, config['learning_rate'], epochs, "gru_bilinear_auxiliary", lambda_val=l_val)
            metrics = evaluate_model_on_split(model, val_loader, device, "gru_bilinear_auxiliary", lambda_val=l_val)
            
            metrics["Seed"] = seed
            metrics["Lambda"] = l_val
            val_records.append(metrics)
            
        # 2. Train baseline Variant B (represented by lambda = 0.0)
        set_seed(seed)
        model_b = EXP026BBilinearSpeakerGRUWithAuxiliary(input_dim, hidden_dim=64).to(device)
        model_b.feature_names = APPROVED_STATE_FREE_FEATURES
        train_model(model_b, train_loader, device, config['learning_rate'], epochs, "gru_bilinear_auxiliary", lambda_val=0.0)
        metrics_b = evaluate_model_on_split(model_b, val_loader, device, "gru_bilinear_auxiliary", lambda_val=0.0)
        metrics_b["Seed"] = seed
        metrics_b["Lambda"] = 0.0
        val_records.append(metrics_b)

    val_df = pd.DataFrame(val_records)
    val_df.to_csv(out_dir / "validation_accuracy_by_lambda.csv", index=False)

    # Global lambda selection (based on mean validation accuracy across seeds)
    lambda_means = val_df[val_df["Lambda"] > 0.0].groupby("Lambda")["Accuracy"].mean().reset_index()
    logger.info("Validation Lambda Performance:\n%s", lambda_means.to_string(index=False))

    # Tie break: highest accuracy, then smaller lambda
    best_idx = lambda_means["Accuracy"].idxmax()
    selected_lambda = float(lambda_means.loc[best_idx, "Lambda"])
    
    # Check if there is a tie
    max_acc = lambda_means["Accuracy"].max()
    tied_lambdas = lambda_means[np.isclose(lambda_means["Accuracy"], max_acc)]["Lambda"].values
    if len(tied_lambdas) > 1:
        selected_lambda = float(np.min(tied_lambdas))
        logger.info(f"Tie detected for validation accuracy ({max_acc:.4f}) between lambdas {tied_lambdas}. Selected smaller lambda: {selected_lambda}")
    else:
        logger.info(f"Selected lambda based on validation accuracy: {selected_lambda}")

    # ==================== STAGE 2: Final Test Evaluation ====================
    logger.info("==================== STAGE 2: FINAL TEST EVALUATION ====================")
    
    seed_metrics = []
    ablation_metrics = []
    ablation_detail_rows = []
    probe_rows = []
    param_rows = []

    for seed in seeds:
        logger.info("========== SEED %d ==========", seed)
        
        # Prepare variants
        set_seed(seed)
        
        # Variant A: Candidate-only scorer
        var_a = EXP026ACandidateOnlyScorer(input_dim, hidden_dim=64).to(device)
        
        # Variant B: Bilinear GRU (no auxiliary weights)
        var_b = EXP026ABilinearSpeakerGRU(input_dim, hidden_dim=64).to(device)
        
        # Variant B2: Bilinear GRU + auxiliary head, lambda = 0
        var_b2 = EXP026BBilinearSpeakerGRUWithAuxiliary(input_dim, hidden_dim=64).to(device)
        
        # Variant C: Bilinear GRU + auxiliary head, validation-selected lambda
        var_c = EXP026BBilinearSpeakerGRUWithAuxiliary(input_dim, hidden_dim=64).to(device)
        
        variants = {
            "nomemory_candidate_only": var_a,
            "gru_bilinear_no_aux": var_b,
            "gru_bilinear_aux_lambda_0": var_b2,
            "gru_bilinear_aux_selected": var_c
        }

        # Train B2 first to use as parameter base, check matches
        for name, model in variants.items():
            model.feature_names = APPROVED_STATE_FREE_FEATURES
            
        # Capacity matches
        var_a_params = sum(p.numel() for p in var_a.parameters() if p.requires_grad)
        var_c_params = sum(p.numel() for p in var_c.parameters() if p.requires_grad)
        
        # Check matching
        if var_c_params > var_a_params * 1.10:
            matched = make_parameter_matched_control(input_dim, var_c_params, base_hidden_dim=64).to(device)
            matched.feature_names = APPROVED_STATE_FREE_FEATURES
            variants["nomemory_parameter_matched"] = matched

        # Exclude W_aux from B2 updates by setting requires_grad to False or verify training equivalence
        # For training B2, we just run train_model with lambda=0.0 (W_aux doesn't get gradients from aux task)
        # To verify shared weight initialization, load var_b weights into B2 and train
        var_b2.load_shared_state_dict(var_b.state_dict())
        
        # Train and evaluate each model on test split
        for var_name, model in variants.items():
            logger.info(f"Training final {var_name}...")
            if var_name == "gru_bilinear_aux_selected":
                train_model(model, train_all_loader, device, config['learning_rate'], epochs, var_name, lambda_val=selected_lambda)
            elif var_name == "gru_bilinear_aux_lambda_0":
                # Train with lambda = 0
                train_model(model, train_all_loader, device, config['learning_rate'], epochs, var_name, lambda_val=0.0)
            else:
                train_model(model, train_all_loader, device, config['learning_rate'], epochs, var_name, lambda_val=0.0)

            # Record Parameter counts
            param_rows.append({
                "Model": var_name,
                "Trainable_Params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "Seed": seed
            })

            # Run evaluation on test split across ablations
            modes = ["normal", "zero_state", "shuffled_update", "teacher_forced_eval_diagnostic"]
            
            # Non-recurrent models only run normal mode
            eval_modes = ["normal"] if var_name.startswith("nomemory") else modes
            
            mode_predictions = {}
            for mode in eval_modes:
                preds = evaluate_scorer(model, test_loader, device, type_mappings, vocab, var_name, mode)
                preds["seed"] = seed
                preds["variant"] = var_name
                preds["mode"] = mode
                
                # Save quote level predictions
                preds.to_csv(out_dir / f"preds_{var_name}_{mode}_seed{seed}.csv", index=False)
                
                mode_predictions[mode] = preds
                
                metrics = compute_overall_metrics(preds)
                metrics["Seed"] = seed
                metrics["Model"] = var_name if mode == "normal" else f"{var_name}_{mode}"
                metrics["Evaluation_Mode"] = mode
                
                if mode == "normal":
                    seed_metrics.append(metrics)
                else:
                    ablation_metrics.append(metrics)

            # Pair predictions for recovery/regression counts
            if var_name in {"gru_bilinear_no_aux", "gru_bilinear_aux_selected"}:
                normal_df = mode_predictions["normal"]
                for mode in ["zero_state", "shuffled_update", "teacher_forced_eval_diagnostic"]:
                    # Compute recovery/regression counts
                    merged = normal_df.merge(mode_predictions[mode], on="quote_id", suffixes=("_normal", "_ablated"))
                    normal_correct = merged["correct_normal"].astype(bool)
                    ablated_correct = merged["correct_ablated"].astype(bool)
                    recoveries = ((~normal_correct) & ablated_correct).sum()
                    regressions = (normal_correct & (~ablated_correct)).sum()
                    
                    ablation_detail_rows.append({
                        "Model": var_name,
                        "Ablation": mode,
                        "Seed": seed,
                        "Recoveries": int(recoveries),
                        "Regressions": int(regressions),
                        "Top1_Agreement": (merged["predicted_candidate_index_normal"] == merged["predicted_candidate_index_ablated"]).mean()
                    })

        # Run probe controls on the final test split using trained recurrent models
        probe_rows.extend(run_probe_controls(var_c, test_seq, test_df_scaled, device, seed))

    # Save final testing outputs
    seed_df = pd.DataFrame(seed_metrics)
    seed_df.to_csv(out_dir / "test_metrics_for_selected_lambda.csv", index=False)
    
    # Compute averaged seed summary
    summary_df = seed_df.groupby("Model").agg(
        Accuracy_Mean=("Accuracy", "mean"),
        Accuracy_Std=("Accuracy", "std"),
        Accuracy_Min=("Accuracy", "min"),
        Accuracy_Max=("Accuracy", "max"),
        Implicit_Mean=("Implicit_Accuracy", "mean"),
        Anaphoric_Mean=("Anaphoric_Accuracy", "mean"),
        MRR_Mean=("MRR", "mean")
    ).reset_index()
    summary_df.to_csv(out_dir / "test_metrics_summary.csv", index=False)
    
    ablation_df = pd.DataFrame(ablation_metrics)
    ablation_df.to_csv(out_dir / "test_metrics_all_lambdas_diagnostic.csv", index=False)
    
    ablation_summary = ablation_df.groupby("Model").agg(
        Accuracy_Mean=("Accuracy", "mean"),
        Implicit_Mean=("Implicit_Accuracy", "mean"),
        Anaphoric_Mean=("Anaphoric_Accuracy", "mean"),
        MRR_Mean=("MRR", "mean")
    ).reset_index()
    ablation_summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    
    pd.DataFrame(ablation_detail_rows).to_csv(out_dir / "prediction_level_ablation_metrics.csv", index=False)
    pd.DataFrame(param_rows).to_csv(out_dir / "parameter_counts.csv", index=False)
    
    probe_df = pd.DataFrame(probe_rows)
    probe_df.to_csv(out_dir / "previous_speaker_probe_controls.csv", index=False)

    # ==================== STAGE 3: Generate Markdown Report ====================
    report = []
    report.append("# EXP026B — Auxiliary Previous-Speaker Supervision Report\n")
    report.append("> EXP026B evaluates whether auxiliary previous-speaker supervision regularizes the GRU hidden state to encode useful history representations that transfer to speaker attribution.\n")
    
    report.append("## Preflight Target Coverage Audit")
    report.append(f"- **Total evaluated quotes**: {preflight_audit['number_of_quotes']}")
    report.append(f"- **Quotes with previous speaker in sequence**: {preflight_audit['quotes_with_previous_speaker']}")
    report.append(f"- **Quotes where previous speaker is present in candidate set**: {preflight_audit['quotes_where_previous_speaker_in_candidate_set']}")
    report.append(f"- **Auxiliary supervision coverage (overall)**: {preflight_audit['auxiliary_supervision_coverage']*100:.2f}%\n")
    
    report.append("## Validation Sweep (Global Selection)")
    report.append(val_df.groupby("Lambda")[["Accuracy", "Attribution_Loss", "Auxiliary_Loss", "Auxiliary_Accuracy", "Auxiliary_MRR", "Auxiliary_PR_AUC"]].mean().reset_index().to_markdown(index=False) + "\n")
    report.append(f"**Selected Lambda**: {selected_lambda}\n")
    
    report.append("## Final Test Results")
    report.append(summary_df.to_markdown(index=False) + "\n")
    
    report.append("## Memory Ablation Results")
    report.append(ablation_summary.to_markdown(index=False) + "\n")
    
    report.append("## Pairwise Ablation Details")
    report.append(pd.DataFrame(ablation_detail_rows).groupby(["Model", "Ablation"])[["Recoveries", "Regressions", "Top1_Agreement"]].mean().reset_index().to_markdown(index=False) + "\n")
    
    report.append("## Previous Speaker Probe Controls")
    report.append(probe_df.groupby("Probe")[["Previous_Speaker_PR_AUC"]].mean().reset_index().to_markdown(index=False) + "\n")
    
    report.append("## Pre-registered Acceptance Criteria")
    report.append("- **Predictive Success**: Selected C_lambda > A by >= +0.5 pp mean accuracy, wins on >= 3/5 seeds.")
    report.append("- **Incremental Success**: Selected C_lambda > B by >= +0.5 pp mean accuracy, wins on >= 3/5 seeds.")
    report.append("- **No Slice Regressions**: Neither implicit nor anaphoric mean accuracy may decrease by more than 0.5 pp relative to A or B.")
    report.append("- **Causal Memory**: Normal - Zero-State >= 0.5 pp, Normal - Shuffled-Update >= 0.25 pp, and Teacher-Forced not worse than Normal by > 0.5 pp.")
    report.append("- **Recovery/Regression**: For C-A and C-B, recoveries must exceed regressions overall.")
    report.append("- **NoMemory Control Clause**: A direct recurrent auxiliary control is not applicable because state-free models have no hidden state h over which cᵀ W_aux h is defined. Parameter controls are instead managed via Variant B2 (lambda=0) and comparison with parameter-matched no-memory baseline.")

    with open(out_dir / "EXP026B_AUXILIARY_PREVIOUS_SPEAKER_REPORT.md", "w") as f:
        f.write("\n".join(report))

    logger.info("EXP026B execution complete. Report written to results/EXP026B/EXP026B_AUXILIARY_PREVIOUS_SPEAKER_REPORT.md")

if __name__ == "__main__":
    main()
