import os
import json
import yaml
import torch
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
from src.neural.models import NoMemoryEntityScorer, RelationalSpeakerGRU
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

def evaluate_scorer(model, dataloader, device, type_mappings, is_gru=False):
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in dataloader:
            features, cids, mask, gold_index, q_ids = batch_to_device(batch, device)
            
            if is_gru:
                scores, _ = model(features, mask, gold_index_for_update=None)
            else:
                scores, _ = model(features, cids, mask, quote_ids=q_ids)
                
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

def main():
    os.makedirs("results/EXP024", exist_ok=True)
    
    with open("configs/EXP021_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = get_device() if config['device'] == 'auto' else config['device']
    logger.info(f"Using device: {device}")
    
    df = load_frozen_exp014_dataset()
    
    df['candidate_position_index'] = df.groupby('quote_id').cumcount()
    df['candidate_position_bucket'] = df['candidate_position_index'].clip(upper=3)
    
    def shuffle_within_group(x):
        return np.random.permutation(x)
        
    df['shuffled_position_index'] = df.groupby('quote_id')['candidate_position_index'].transform(shuffle_within_group)
    df['shuffled_position_bucket'] = df['shuffled_position_index'].clip(upper=3)
    
    vocab = build_character_vocab(df, vocab_path="results/EXP024/character_vocab.json")
    
    APPROVED_STATE_FREE_FEATURES = [
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
    
    missing = sorted(set(APPROVED_STATE_FREE_FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Missing approved features: {missing}")

    conditions = {
        'nomemory_baseline': {
            'features': APPROVED_STATE_FREE_FEATURES,
            'is_gru': False
        },
        'nomemory_plus_position_index': {
            'features': APPROVED_STATE_FREE_FEATURES + ["candidate_position_index"],
            'is_gru': False
        },
        'nomemory_plus_position_bucket': {
            'features': APPROVED_STATE_FREE_FEATURES + ["candidate_position_bucket"],
            'is_gru': False
        },
        'nomemory_plus_shuffled_position_index': {
            'features': APPROVED_STATE_FREE_FEATURES + ["shuffled_position_index"],
            'is_gru': False
        },
        'nomemory_plus_shuffled_position_bucket': {
            'features': APPROVED_STATE_FREE_FEATURES + ["shuffled_position_bucket"],
            'is_gru': False
        },
        'gru_baseline': {
            'features': APPROVED_STATE_FREE_FEATURES,
            'is_gru': True
        },
        'gru_plus_position_index': {
            'features': APPROVED_STATE_FREE_FEATURES + ["candidate_position_index"],
            'is_gru': True
        },
        'gru_plus_position_bucket': {
            'features': APPROVED_STATE_FREE_FEATURES + ["candidate_position_bucket"],
            'is_gru': True
        }
    }
    
    with open("results/EXP024/feature_list.json", "w") as f:
        json.dump(
            {
                "approved_features": APPROVED_STATE_FREE_FEATURES,
                "conditions": {k: v['features'] for k, v in conditions.items()}
            },
            f, indent=4
        )
        
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    epochs = config['epochs']
    is_cpu_test = os.environ.get("CPU_TEST_RUN") == "1"
    if is_cpu_test:
        epochs = 1
        
    run_config = {
        "seed": 12345,
        "epochs": epochs,
        "learning_rate": config["learning_rate"],
        "cpu_test_run": is_cpu_test,
    }
        
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
                
    metrics_summary = {}
    all_preds = {}
    
    for mode, mode_config in conditions.items():
        set_seed(12345)
        logger.info(f"--- Setting up {mode} ---")
        
        feature_cols = mode_config['features']
        is_gru = mode_config['is_gru']
        
        # Scaling
        scaler = StandardScaler()
        train_df_scaled = train_df.copy()
        test_df_scaled = test_df.copy()
        
        train_df_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
        
        train_seq = TensorSequenceDataset(train_df_scaled, feature_cols, feature_mode='all', vocab=vocab, scaler=None)
        test_seq = TensorSequenceDataset(test_df_scaled, feature_cols, feature_mode='all', vocab=vocab, scaler=None)
        
        def collate_fn(batch):
            return batch[0]
            
        train_loader = DataLoader(train_seq, batch_size=1, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_seq, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        input_dim = len(feature_cols)
        
        if not is_gru:
            model = NoMemoryEntityScorer(
                feature_dim=input_dim, vocab_size=len(vocab), emb_dim=32, hidden_dim=64, anchor_mode='no_anchor'
            ).to(device)
        else:
            model = RelationalSpeakerGRU(
                feature_dim=input_dim, hidden_dim=64
            ).to(device)
            
        opt_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(opt_params, lr=config['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        logger.info(f"--- Training {mode} ---")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_items = 0
            
            for batch in train_loader:
                features, cids, mask, gold_index, q_ids = batch_to_device(batch, device)
                optimizer.zero_grad()
                
                if is_gru:
                    scores, _ = model(features, mask, gold_index_for_update=gold_index)
                else:
                    scores, _ = model(features, cids, mask, quote_ids=q_ids)
                    
                scores = scores.squeeze(0)
                loss = criterion(scores, gold_index.squeeze(0) if gold_index.dim() == 2 else gold_index)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(gold_index)
                total_items += len(gold_index)
                
            logger.info(f"{mode} - Epoch {epoch+1}/{epochs} - Loss: {total_loss/total_items:.4f}")
            
        logger.info(f"Evaluating {mode}...")
        preds = evaluate_scorer(model, test_loader, device, type_mappings, is_gru=is_gru)
        preds.to_csv(f"results/EXP024/preds_{mode}.csv", index=False)
        all_preds[mode] = preds
        
        mets = compute_metrics(preds)
        metrics_summary[mode] = mets

    # Generate Report
    report = ["# EXP024 Position Feature Experiment\n"]
    
    report.append("## Core Metrics")
    report.append("| Condition | Overall Acc | Implicit Acc | Anaphoric Acc | MRR | LogLoss |")
    report.append("|-----------|-------------|--------------|---------------|-----|---------|")
    for mode in conditions.keys():
        mets = metrics_summary[mode]
        report.append(f"| {mode} | {mets['Accuracy']*100:.2f}% | {mets['Implicit_Accuracy']*100:.2f}% | {mets['Anaphoric_Accuracy']*100:.2f}% | {mets['MRR']:.4f} | {mets['LogLoss']:.4f} |")
        
    report.append("\n## McNemar Statistical Tests")
    
    stat_tests = {}
    def add_test(name, mode_a, mode_b):
        if mode_a not in all_preds or mode_b not in all_preds: return
        pval = mcnemar_test(all_preds[mode_a], all_preds[mode_b])
        report.append(f"- **{name} ({mode_a} vs {mode_b})**: {pval:.4e}")
        stat_tests[name] = pval
        
    add_test("nomemory_plus_position_index vs nomemory_baseline", "nomemory_plus_position_index", "nomemory_baseline")
    add_test("nomemory_plus_position_bucket vs nomemory_baseline", "nomemory_plus_position_bucket", "nomemory_baseline")
    add_test("gru_plus_position_index vs gru_baseline", "gru_plus_position_index", "gru_baseline")
    add_test("gru_plus_position_bucket vs gru_baseline", "gru_plus_position_bucket", "gru_baseline")
    add_test("nomemory_plus_shuffled_position_index vs nomemory_baseline", "nomemory_plus_shuffled_position_index", "nomemory_baseline")

    with open("results/EXP024/statistical_tests.json", "w") as f:
        json.dump(stat_tests, f, indent=4)
        
    # Decision Rules Evaluation
    report.append("\n## Decision Rule Interpretation")
    
    def check_gain(mode_target, mode_base):
        if mode_target not in metrics_summary or mode_base not in metrics_summary: return False, False, 0
        diff = metrics_summary[mode_target]['Accuracy'] - metrics_summary[mode_base]['Accuracy']
        pval = stat_tests.get(f"{mode_target} vs {mode_base}", 1.0)
        return (diff >= 0.01), (pval < 0.05), diff

    nm_idx_1pp, nm_idx_sig, nm_idx_diff = check_gain("nomemory_plus_position_index", "nomemory_baseline")
    nm_bkt_1pp, nm_bkt_sig, nm_bkt_diff = check_gain("nomemory_plus_position_bucket", "nomemory_baseline")
    gru_idx_1pp, gru_idx_sig, gru_idx_diff = check_gain("gru_plus_position_index", "gru_baseline")
    gru_bkt_1pp, gru_bkt_sig, gru_bkt_diff = check_gain("gru_plus_position_bucket", "gru_baseline")
    
    position_helps = (nm_idx_1pp and nm_idx_sig) or (nm_bkt_1pp and nm_bkt_sig) or (gru_idx_1pp and gru_idx_sig) or (gru_bkt_1pp and gru_bkt_sig)
    
    if position_helps:
        if nm_idx_1pp and not gru_idx_1pp:
            report.append("> **Conclusion**: Position helps No-Memory but not GRU. GRU already captures some salience structure natively.")
        else:
            report.append("> **Conclusion**: Position improves accuracy by >=1pp significantly. **However**, if the audit confirmed ordering is arbitrary, report this as a frozen dataset artifact and DO NOT accept it as a principled feature.")
    else:
        report.append("> **Conclusion**: Position does NOT help significantly after being explicitly represented. The previous EXP023 gain was architecture-specific.")
        
    with open("results/EXP024/metrics_report.md", "w") as f:
        f.write("\n".join(report))
        
    logger.info("EXP024B execution complete.")

if __name__ == "__main__":
    main()
