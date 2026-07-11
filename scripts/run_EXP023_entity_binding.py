import os
import json
import yaml
import torch
import hashlib
import logging
import random
import pandas as pd
import numpy as np
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from src.utils.device import get_device
from src.evaluation.runner import load_frozen_exp014_dataset
from src.neural.sequence_dataset import build_character_vocab, TensorSequenceDataset
from src.neural.models import NoMemoryEntityScorer
from scripts.run_EXP021A_2_mlp_ce import compute_metrics, mcnemar_test

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    )

def parameter_trainability(model):
    return {
        name: {
            "requires_grad": bool(param.requires_grad),
            "shape": list(param.shape),
        }
        for name, param in model.named_parameters()
    }

def evaluate_scorer(model, dataloader, device, type_mappings):
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in dataloader:
            features, cids, mask, gold_index, q_ids = batch_to_device(batch, device)
            
            scores, anchors = model(features, cids, mask, quote_ids=q_ids)
            scores = scores.squeeze(0)
            
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

def get_anchor_persistence_score(model, dataloader, device):
    """
    Computes cosine similarity between anchor vectors for the same entity
    across successive observed appearances in the evaluated sequence.
    """
    model.eval()
    cos_sims = []
    
    with torch.no_grad():
        for batch in dataloader:
            features, cids, mask, _, q_ids = batch_to_device(batch, device)
            
            _, anchors = model(features, cids, mask, quote_ids=q_ids)
            anchors = anchors.squeeze(0) # [seq_len, max_cand, emb_dim]
            
            seq_len, max_cand, _ = anchors.shape
            
            # Map entity_id -> last seen anchor
            last_anchor = {}
            for t in range(seq_len):
                for c in range(max_cand):
                    if mask[0, t, c] if mask.dim() == 3 else mask[t, c]:
                        eid = cids[0, t, c].item() if cids.dim() == 3 else cids[t, c].item()
                        anchor_t = anchors[t, c]
                        
                        if eid in last_anchor:
                            prev = last_anchor[eid]
                            sim = torch.nn.functional.cosine_similarity(anchor_t.unsqueeze(0), prev.unsqueeze(0)).item()
                            cos_sims.append(sim)
                            
                        last_anchor[eid] = anchor_t
                        
    if len(cos_sims) == 0:
        return 0.0
    return np.mean(cos_sims)

def get_unique_anchors(model, dataloader, device):
    model.eval()
    unique_hashes = set()
    with torch.no_grad():
        for batch in dataloader:
            features, cids, mask, _, q_ids = batch_to_device(batch, device)
            
            _, anchors = model(features, cids, mask, quote_ids=q_ids)
            anchors = anchors.squeeze(0)
            seq_len, max_cand, _ = anchors.shape
            for t in range(seq_len):
                for c in range(max_cand):
                    if mask[0, t, c] if mask.dim() == 3 else mask[t, c]:
                        vec = anchors[t, c].cpu().numpy().tobytes()
                        unique_hashes.add(vec)
    return len(unique_hashes)

def main():
    os.makedirs("results/EXP023", exist_ok=True)
    
    with open("configs/EXP021_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = get_device() if config['device'] == 'auto' else config['device']
    logger.info(f"Using device: {device}")
    
    df = load_frozen_exp014_dataset()
    vocab = build_character_vocab(df, vocab_path="results/EXP023/character_vocab.json")
    
    # 1. Exact Approved EXP021A2 State-Free Feature list
    APPROVED_EXP021A2_STATE_FREE_FEATURES = [
        "candidate_in_quote_chain",
        "candidate_is_attributed_speaker",
        "candidate_is_explicit_mention",
        "candidate_is_recent_mention",
        "chain_recency",
        "conversation_length",
        "conversation_turn_index",
        "discourse_context_length",
        "discourse_dialogue_position",
        "lexical_has_exclamation",
        "lexical_has_question_mark",
        "lexical_quote_length_chars",
        "lexical_quote_length_tokens",
        "nearest_coref_dist",
        "recent_mention_count",
    ]
    
    missing = sorted(set(APPROVED_EXP021A2_STATE_FREE_FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Missing approved EXP021A.2 features: {missing}")

    state_free_cols = APPROVED_EXP021A2_STATE_FREE_FEATURES
    
    # 2. Fail-fast assertions against forbidden features
    FORBIDDEN_FEATURES = {
        "candidate_is_last_speaker",
        "candidate_is_previous_speaker",
        "candidate_in_participant_stack",
        "candidate_stack_depth",
        "conversation_speaker_change",
        "conv_active_id",
        "conv_interruption_distance",
        "quote_start_byte",
        "quote_end_byte",
        "quoteByteSpans",
        "symbolic_explicit_rule_fired",
        "symbolic_alternation_rule_fired",
    }

    present_forbidden = sorted(set(state_free_cols) & FORBIDDEN_FEATURES)
    if present_forbidden:
        raise ValueError(f"Forbidden EXP023 features present: {present_forbidden}")

    for c in state_free_cols:
        if c.startswith("symbolic_"):
            raise ValueError(f"Forbidden symbolic feature present: {c}")
            
    # 3. Save feature list
    with open("results/EXP023/feature_list.json", "w") as f:
        json.dump(
            {
                "approved_feature_count": len(state_free_cols),
                "approved_features": state_free_cols,
                "forbidden_features_absent": True,
                "source": "EXP021A.2 state-free feature list",
            },
            f,
            indent=4,
        )
        
    logger.info(f"Using {len(state_free_cols)} state-free features: {state_free_cols}")
    
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    # 4. Save feature-equivalence audit
    feature_audit = {
        "approved_features": state_free_cols,
        "feature_count": len(state_free_cols),
        "dataset_hash": str(pd.util.hash_pandas_object(df, index=True).sum()),
        "train_quote_count": int(train_df["quote_id"].nunique()),
        "test_quote_count": int(test_df["quote_id"].nunique()),
        "train_candidate_count": int(len(train_df)),
        "test_candidate_count": int(len(test_df)),
        "forbidden_features_checked": sorted(FORBIDDEN_FEATURES),
        "forbidden_features_present": present_forbidden,
    }

    with open("results/EXP023/feature_equivalence_audit.json", "w") as f:
        json.dump(feature_audit, f, indent=4)
        
    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    
    train_df_scaled[state_free_cols] = scaler.fit_transform(train_df[state_free_cols])
    test_df_scaled[state_free_cols] = scaler.transform(test_df[state_free_cols])
    
    train_seq = TensorSequenceDataset(train_df_scaled, state_free_cols, feature_mode='all', vocab=vocab, scaler=None)
    test_seq = TensorSequenceDataset(test_df_scaled, state_free_cols, feature_mode='all', vocab=vocab, scaler=None)
    
    def collate_fn(batch):
        return batch[0]
        
    train_loader = DataLoader(train_seq, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_seq, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    input_dim = train_seq[0].candidate_features.shape[-1]
    
    # Precompute pretrained embeddings
    frozen_embs = torch.randn(len(vocab), 32, generator=torch.Generator().manual_seed(42))
    frozen_embs[0] = 0
    
    hash_embs = torch.randn(len(vocab), 32)
    for entity, idx in vocab.items():
        if idx == 0:
            hash_embs[idx] = 0
            continue
        seed_str = f"hash_{entity}"
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
        gen = torch.Generator(device='cpu').manual_seed(seed)
        hash_embs[idx] = torch.randn(32, generator=gen)
        
    conditions = {
        'no_anchor': None,
        'constant': None,
        'position': None,
        'ephemeral': None,
        'unstable': None,
        'frozen_persistent': frozen_embs,
        'deterministic_hash': hash_embs,
        'trainable_persistent': None,
        'shuffled_persistent': frozen_embs # use same frozen persistent embs for shuffled persistent, just permuted (shuffled logic is in model)
    }
    
    epochs = config['epochs']
    is_cpu_test = os.environ.get("CPU_TEST_RUN") == "1"
    if is_cpu_test:
        epochs = 1
        
    run_config = {
        "seed": 12345,
        "epochs": epochs,
        "learning_rate": config["learning_rate"],
        "input_dim": int(input_dim),
        "anchor_dim": 32,
        "hidden_dim": 64,
        "conditions": list(conditions.keys()),
        "cpu_test_run": is_cpu_test,
    }

    with open("results/EXP023/run_config.json", "w") as f:
        json.dump(run_config, f, indent=4)
        
    # Quote mappings
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
                
    diagnostics = {}
    metrics_summary = {}
    all_preds = {}
    
    for mode, pretrained_emb in conditions.items():
        # Reset seed per condition
        set_seed(12345)
        
        logger.info(f"--- Training {mode} ---")
        model = NoMemoryEntityScorer(
            feature_dim=input_dim, 
            vocab_size=len(vocab),
            emb_dim=32,
            hidden_dim=64,
            anchor_mode=mode,
            pretrained_emb=pretrained_emb
        ).to(device)
        
        # Safeguards for param trainability
        if mode in {"frozen_persistent", "deterministic_hash", "shuffled_persistent", "constant"}:
            for name, param in model.named_parameters():
                if "char_emb" in name or "constant_vector" in name:
                    assert not param.requires_grad, (
                        f"{mode} should have frozen anchor parameters, "
                        f"but {name} is trainable."
                    )
        
        if mode == "trainable_persistent":
            has_trainable_anchor = any(
                param.requires_grad
                for name, param in model.named_parameters()
                if "char_emb" in name
            )
            assert has_trainable_anchor, "trainable_persistent has no trainable character embedding."
        
        # Only optimizer parameters that require gradients
        opt_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(opt_params, lr=config['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_items = 0
            
            for batch in train_loader:
                features, cids, mask, gold_index, q_ids = batch_to_device(batch, device)
                
                optimizer.zero_grad()
                
                scores, _ = model(features, cids, mask, quote_ids=q_ids)
                scores = scores.squeeze(0)
                loss = criterion(scores, gold_index)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(gold_index)
                total_items += len(gold_index)
                
            logger.info(f"{mode} - Epoch {epoch+1}/{epochs} - Loss: {total_loss/total_items:.4f}")
            
        logger.info(f"Evaluating {mode}...")
        preds = evaluate_scorer(model, test_loader, device, type_mappings)
        preds.to_csv(f"results/EXP023/preds_{mode}.csv", index=False)
        all_preds[mode] = preds
        
        mets = compute_metrics(preds)
        metrics_summary[mode] = mets
        
        if mode == "no_anchor":
            no_anchor_acc = mets["Accuracy"]
            no_anchor_impl = mets["Implicit_Accuracy"]
        
            if not (0.66 <= no_anchor_acc <= 0.73):
                raise RuntimeError(
                    f"Unexpected no_anchor accuracy: {no_anchor_acc:.4f}. "
                    "Expected roughly 0.69-0.70 based on EXP021A.2 clean feature audit. "
                    "Possible feature leakage or training mismatch."
                )
        
            if not (0.55 <= no_anchor_impl <= 0.66):
                raise RuntimeError(
                    f"Unexpected no_anchor implicit accuracy: {no_anchor_impl:.4f}. "
                    "Expected roughly 0.60-0.61 based on clean feature audit."
                )
        
        persistence = get_anchor_persistence_score(model, test_loader, device)
        n_anchors = get_unique_anchors(model, test_loader, device)
        
        diagnostics[mode] = {
            'anchor_persistence_score': float(persistence),
            'unique_anchors': n_anchors,
            'trainable_params': parameter_trainability(model)
        }
        
    with open("results/EXP023/anchor_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=4)
        
    # Generate Report
    report = ["# EXP023 Entity Binding Analysis\n"]
    
    report.append("## Core Metrics")
    report.append("| Condition | Overall Acc | Implicit Acc | Anaphoric Acc | MRR | LogLoss |")
    report.append("|-----------|-------------|--------------|---------------|-----|---------|")
    for mode in conditions.keys():
        mets = metrics_summary[mode]
        report.append(f"| {mode} | {mets['Accuracy']*100:.2f}% | {mets['Implicit_Accuracy']*100:.2f}% | {mets['Anaphoric_Accuracy']*100:.2f}% | {mets['MRR']:.4f} | {mets['LogLoss']:.4f} |")
        
    report.append("\n## Anchor Diagnostics")
    report.append("| Condition | Persistence Score | Unique Anchors | Trainable Params |")
    report.append("|-----------|-------------------|----------------|------------------|")
    for mode, diag in diagnostics.items():
        trainable_str = ", ".join([f"{k}: {'grad' if v['requires_grad'] else 'frozen'}" for k, v in diag['trainable_params'].items()])
        report.append(f"| {mode} | {diag['anchor_persistence_score']:.4f} | {diag['unique_anchors']} | {trainable_str} |")
        
    report.append("\n## McNemar Statistical Tests")
    
    # Load MLP CE
    mlp_path = "results/EXP021A_2/predictions.csv"
    if os.path.exists(mlp_path):
        mlp_preds = pd.read_csv(mlp_path)
    else:
        mlp_preds = None
        
    report.append("### Vs Frozen Persistent Random Anchor")
    frozen_preds = all_preds['frozen_persistent']
    for mode in conditions.keys():
        if mode == 'frozen_persistent': continue
        pval = mcnemar_test(all_preds[mode], frozen_preds)
        report.append(f"- **{mode}**: {pval:.4e}")
        
    report.append("\n### Vs No Anchor Baseline")
    no_anchor_preds = all_preds['no_anchor']
    for mode in conditions.keys():
        if mode == 'no_anchor': continue
        pval = mcnemar_test(all_preds[mode], no_anchor_preds)
        report.append(f"- **{mode}**: {pval:.4e}")
        
    if mlp_preds is not None:
        report.append("\n### Vs MLP CE Baseline (EXP021A.2)")
        for mode in conditions.keys():
            pval = mcnemar_test(all_preds[mode], mlp_preds)
            report.append(f"- **{mode}**: {pval:.4e}")
            
    # Success criterion: Frozen Persistent Random beats Ephemeral by 1.5 pp
    frozen_acc = metrics_summary['frozen_persistent']['Accuracy'] * 100
    ephem_acc = metrics_summary['ephemeral']['Accuracy'] * 100
    diff = frozen_acc - ephem_acc
    report.append("\n## Success Criteria")
    report.append(f"**Frozen Persistent - Ephemeral >= 1.5 pp**: {'PASS' if diff >= 1.5 else 'FAIL'} ({diff:.2f} pp)")
    
    with open("results/EXP023/metrics_report.md", "w") as f:
        f.write("\n".join(report))
        
    logger.info("EXP023 execution complete.")

if __name__ == "__main__":
    main()
