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
from src.neural.models import NoMemoryEntityScorer, RelationalSpeakerGRU, EntityAnchoredRelationalGRU
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

def generate_model_summary(model, condition_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    
    anchor_related_names = [n for n, p in model.named_parameters() if 'char_emb' in n or 'pos_emb' in n or 'constant' in n]
    anchor_frozen = all(not param.requires_grad for name, param in model.named_parameters() if name in anchor_related_names)
    
    return {
        "condition": condition_name,
        "class_name": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_parameter_names": trainable_param_names,
        "anchor_related_parameter_names": anchor_related_names,
        "anchor_parameters_frozen": anchor_frozen
    }

def evaluate_scorer(model, dataloader, device, type_mappings, is_gru=False):
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in dataloader:
            features, cids, mask, gold_index, q_ids = batch_to_device(batch, device)
            
            if is_gru:
                if isinstance(model, RelationalSpeakerGRU):
                    scores, _ = model(features, mask, gold_index_for_update=None)
                else:
                    scores, _ = model(features, cids, mask, gold_index_for_update=None)
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
    os.makedirs("results/EXP023B", exist_ok=True)
    
    with open("configs/EXP021_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = get_device() if config['device'] == 'auto' else config['device']
    logger.info(f"Using device: {device}")
    
    df = load_frozen_exp014_dataset()
    vocab = build_character_vocab(df, vocab_path="results/EXP023B/character_vocab.json")
    
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
        raise ValueError(f"Missing approved EXP023B features: {missing}")

    state_free_cols = APPROVED_STATE_FREE_FEATURES
    
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
        raise ValueError(f"Forbidden EXP023B features present: {present_forbidden}")

    for c in state_free_cols:
        if c.startswith("symbolic_"):
            raise ValueError(f"Forbidden symbolic feature present: {c}")
            
    with open("results/EXP023B/feature_list.json", "w") as f:
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
    
    # Precompute pretrained embeddings exactly like EXP023
    frozen_embs = torch.randn(len(vocab), 32, generator=torch.Generator().manual_seed(42))
    frozen_embs[0] = 0
    
    conditions = [
        'nomemory_no_anchor',
        'nomemory_persistent_anchor',
        'gru_no_anchor',
        'gru_persistent_anchor'
    ]
    
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
        "conditions": conditions,
        "cpu_test_run": is_cpu_test,
        "model_summaries": {}
    }
        
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
                
    metrics_summary = {}
    all_preds = {}
    
    for mode in conditions:
        set_seed(12345)
        logger.info(f"--- Setting up {mode} ---")
        
        if mode == 'nomemory_no_anchor':
            model = NoMemoryEntityScorer(
                feature_dim=input_dim, vocab_size=len(vocab), emb_dim=32, hidden_dim=64, anchor_mode='no_anchor'
            ).to(device)
            is_gru = False
        elif mode == 'nomemory_persistent_anchor':
            model = NoMemoryEntityScorer(
                feature_dim=input_dim, vocab_size=len(vocab), emb_dim=32, hidden_dim=64, anchor_mode='frozen_persistent', pretrained_emb=frozen_embs
            ).to(device)
            is_gru = False
        elif mode == 'gru_no_anchor':
            model = RelationalSpeakerGRU(
                feature_dim=input_dim, hidden_dim=64
            ).to(device)
            is_gru = True
        elif mode == 'gru_persistent_anchor':
            model = EntityAnchoredRelationalGRU(
                feature_dim=input_dim, vocab_size=len(vocab), emb_dim=32, hidden_dim=64
            ).to(device)
            model.char_emb.weight.data.copy_(frozen_embs)
            model.char_emb.weight.requires_grad = False
            is_gru = True
            
        # Specific Assertions per condition
        if mode == 'gru_no_anchor':
            has_char_emb = any('char_emb' in n or 'pos_emb' in n or 'constant' in n for n, _ in model.named_parameters())
            assert not has_char_emb, "gru_no_anchor should not have any anchor embeddings instantiated."
        
        if mode == 'gru_persistent_anchor':
            assert not model.char_emb.weight.requires_grad, "gru_persistent_anchor char_emb MUST be frozen!"
            
        if mode == 'nomemory_persistent_anchor':
            assert not model.char_emb.weight.requires_grad, "nomemory_persistent_anchor char_emb MUST be frozen!"
            
        # Extract model summary
        summary = generate_model_summary(model, mode)
        run_config["model_summaries"][mode] = summary
        
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
                    if isinstance(model, RelationalSpeakerGRU):
                        scores, _ = model(features, mask, gold_index_for_update=gold_index)
                    else:
                        scores, _ = model(features, cids, mask, gold_index_for_update=gold_index)
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
        preds.to_csv(f"results/EXP023B/preds_{mode}.csv", index=False)
        all_preds[mode] = preds
        
        mets = compute_metrics(preds)
        metrics_summary[mode] = mets

    # Save run config at the end so it includes summaries
    with open("results/EXP023B/run_config.json", "w") as f:
        json.dump(run_config, f, indent=4)
        
    # Generate Report
    report = ["# EXP023B Entity Binding Interaction Analysis\n"]
    
    report.append("## Core Metrics")
    report.append("| Condition | Overall Acc | Implicit Acc | Anaphoric Acc | MRR | LogLoss |")
    report.append("|-----------|-------------|--------------|---------------|-----|---------|")
    for mode in conditions:
        mets = metrics_summary[mode]
        report.append(f"| {mode} | {mets['Accuracy']*100:.2f}% | {mets['Implicit_Accuracy']*100:.2f}% | {mets['Anaphoric_Accuracy']*100:.2f}% | {mets['MRR']:.4f} | {mets['LogLoss']:.4f} |")
        
    report.append("\n## McNemar Statistical Tests")
    
    stat_tests = {}
    
    def add_test(name, mode_a, mode_b):
        if mode_a not in all_preds or mode_b not in all_preds: return
        pval = mcnemar_test(all_preds[mode_a], all_preds[mode_b])
        report.append(f"- **{name} ({mode_a} vs {mode_b})**: {pval:.4e}")
        stat_tests[name] = pval
        
    add_test("GRU Persistent Anchor vs GRU No Anchor", "gru_persistent_anchor", "gru_no_anchor")
    add_test("GRU Persistent Anchor vs Nomemory Persistent Anchor", "gru_persistent_anchor", "nomemory_persistent_anchor")
    add_test("GRU Persistent Anchor vs Nomemory No Anchor", "gru_persistent_anchor", "nomemory_no_anchor")
    add_test("Nomemory Persistent Anchor vs Nomemory No Anchor", "nomemory_persistent_anchor", "nomemory_no_anchor")
    add_test("GRU No Anchor vs Nomemory No Anchor", "gru_no_anchor", "nomemory_no_anchor")

    with open("results/EXP023B/statistical_tests.json", "w") as f:
        json.dump(stat_tests, f, indent=4)
        
    # Decision Rules evaluation
    report.append("\n## Decision Rule Interpretation")
    
    gru_pers = metrics_summary['gru_persistent_anchor']['Accuracy']
    gru_no = metrics_summary['gru_no_anchor']['Accuracy']
    nom_pers = metrics_summary['nomemory_persistent_anchor']['Accuracy']
    nom_no = metrics_summary['nomemory_no_anchor']['Accuracy']
    
    if gru_pers > gru_no and gru_pers > nom_pers and gru_pers > nom_no:
        if stat_tests["GRU Persistent Anchor vs GRU No Anchor"] < 0.05 and stat_tests["GRU Persistent Anchor vs Nomemory Persistent Anchor"] < 0.05 and stat_tests["GRU Persistent Anchor vs Nomemory No Anchor"] < 0.05:
            report.append("> **Conclusion**: Persistent anchors become useful only with recurrent memory.")
        else:
            report.append("> **Conclusion**: The trend supports GRU-anchor interaction, but results are not fully statistically significant across all bounds.")
    elif abs(gru_pers - gru_no) < 0.005: # less than 0.5% diff
        report.append("> **Conclusion**: GRU memory helps or fails independently of anchors. No anchor-memory interaction.")
    elif abs(nom_pers - nom_no) < 0.005 and gru_pers > gru_no:
        report.append("> **Conclusion**: This supports the anchor-memory interaction hypothesis.")
    elif gru_pers <= nom_no:
        report.append("> **Conclusion**: The GRU-anchor architecture is not justified.")
    else:
        report.append("> **Conclusion**: Mixed outcome requiring manual review.")
    
    with open("results/EXP023B/metrics_report.md", "w") as f:
        f.write("\n".join(report))
        
    logger.info("EXP023B execution complete.")

if __name__ == "__main__":
    main()
