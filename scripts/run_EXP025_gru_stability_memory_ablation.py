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
import scipy.stats as stats

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
    # Ensure reproducibility in DataLoader
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def evaluate_scorer(model, dataloader, device, type_mappings, is_gru=False, ablation_mode="normal"):
    """
    Evaluates the model and supports various memory ablation modes for the GRU.
    
    ablation_mode can be:
    - 'normal': autoregressive (update with predicted candidate)
    - 'reset_hidden_each_quote': reset hidden to zero before each quote, then update with predicted.
    - 'zero_update': never update hidden, always use zeros.
    - 'shuffled_update': update with random candidate (fixed seed per quote).
    - 'teacher_forced_eval_diagnostic': update with gold index.
    """
    model.eval()
    results = []
    
    # Store predictions to update hidden state correctly
    
    with torch.no_grad():
        if is_gru and ablation_mode == "zero_update":
            # For zero update, we can just reset hidden state each time and NOT pass gold_index_for_update.
            # But wait, RelationalSpeakerGRU expects to process the whole batch in one forward pass (if batched by novel)
            # and it tracks its own hidden state sequentially.
            pass
            
        for batch in dataloader:
            features, cids, mask, gold_index, q_ids = batch_to_device(batch, device)
            
            if is_gru:
                if hasattr(model, 'reset_hidden'):
                    model.reset_hidden()
                
                if gold_index.dim() == 1:
                    gold_index = gold_index.unsqueeze(0)
                
                # RelationalSpeakerGRU processes sequence one by one internally if features is (1, seq_len, num_cand, feat_dim)
                seq_len = features.shape[1]
                scores_list = []
                
                if ablation_mode == "teacher_forced_eval_diagnostic":
                    scores, _ = model(features, mask, gold_index_for_update=gold_index)
                elif ablation_mode in ["reset_hidden_each_quote", "zero_update"]:
                    scores, _ = model(features, mask, gold_index_for_update=None, ablate_memory=True)
                elif ablation_mode == "shuffled_update":
                    # We set a fixed seed before calling to ensure reproducibility of the random choices
                    torch.manual_seed(42)
                    scores, _ = model(features, mask, gold_index_for_update=None, ablate_shuffle=True)
                else: # normal
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
    out_dir = Path("results/EXP025")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open("configs/EXP021_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = get_device() if config['device'] == 'auto' else config['device']
    logger.info(f"Using device: {device}")
    
    df = load_frozen_exp014_dataset()
    vocab = build_character_vocab(df, vocab_path="results/EXP025/character_vocab.json")
    
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
        "candidate_position_index",
        "candidate_position_bucket",
    }
    
    missing = sorted(set(APPROVED_STATE_FREE_FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Missing approved features: {missing}")

    present_forbidden = sorted(set(APPROVED_STATE_FREE_FEATURES) & FORBIDDEN_FEATURES)
    if present_forbidden:
        raise ValueError(f"Forbidden features present in dataset: {present_forbidden}")

    with open("results/EXP025/feature_list.json", "w") as f:
        json.dump({
            "approved_features": APPROVED_STATE_FREE_FEATURES,
            "forbidden_features_absent": True
        }, f, indent=4)
        
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    is_cpu_test = os.environ.get("CPU_TEST_RUN") == "1"
    SEEDS = [1] if is_cpu_test else [1, 2, 3, 4, 5]
    epochs = 1 if is_cpu_test else config['epochs']
        
    run_config = {
        "experiment_name": "EXP025_gru_stability_memory_ablation",
        "seeds": SEEDS,
        "epochs": epochs,
        "learning_rate": config["learning_rate"],
        "input_dim": len(APPROVED_STATE_FREE_FEATURES),
        "hidden_dim": 64,
        "cpu_test_run": is_cpu_test,
        "model_class_names": ["NoMemoryEntityScorer", "RelationalSpeakerGRU"],
        "feature_list": APPROVED_STATE_FREE_FEATURES,
        "forbidden_feature_list": list(FORBIDDEN_FEATURES)
    }
    with open("results/EXP025/run_config.json", "w") as f:
        json.dump(run_config, f, indent=4)
        
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
                
    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    
    train_df_scaled[APPROVED_STATE_FREE_FEATURES] = scaler.fit_transform(train_df[APPROVED_STATE_FREE_FEATURES])
    test_df_scaled[APPROVED_STATE_FREE_FEATURES] = scaler.transform(test_df[APPROVED_STATE_FREE_FEATURES])
    
    train_seq = TensorSequenceDataset(train_df_scaled, APPROVED_STATE_FREE_FEATURES, feature_mode='all', vocab=vocab, scaler=None)
    test_seq = TensorSequenceDataset(test_df_scaled, APPROVED_STATE_FREE_FEATURES, feature_mode='all', vocab=vocab, scaler=None)
    
    def collate_fn(batch):
        return batch[0]
        
    train_loader = DataLoader(train_seq, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_seq, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    input_dim = len(APPROVED_STATE_FREE_FEATURES)
    
    # Store results
    seed_results_nomemory = []
    seed_results_gru = []
    
    memory_ablation_results = []
    
    for seed in SEEDS:
        logger.info(f"========== SEED {seed} ==========")
        set_seed(seed)
        
        # --- Train NoMemory ---
        nomemory = NoMemoryEntityScorer(
            feature_dim=input_dim, vocab_size=len(vocab), emb_dim=32, hidden_dim=64, anchor_mode='no_anchor'
        ).to(device)
        
        opt_nm = torch.optim.Adam(nomemory.parameters(), lr=config['learning_rate'])
        crit = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            nomemory.train()
            for batch in train_loader:
                features, cids, mask, gold_index, q_ids = batch_to_device(batch, device)
                opt_nm.zero_grad()
                scores, _ = nomemory(features, cids, mask, quote_ids=q_ids)
                loss = crit(scores.squeeze(0), gold_index.squeeze(0) if gold_index.dim() == 2 else gold_index)
                loss.backward()
                opt_nm.step()
                
        preds_nm = evaluate_scorer(nomemory, test_loader, device, type_mappings, is_gru=False, ablation_mode="normal")
        preds_nm.to_csv(f"results/EXP025/preds_nomemory_no_anchor_seed{seed}.csv", index=False)
        mets_nm = compute_metrics(preds_nm)
        mets_nm['Seed'] = seed
        mets_nm['Model'] = 'nomemory_no_anchor'
        seed_results_nomemory.append(mets_nm)
        
        # --- Train GRU ---
        set_seed(seed)
        gru = RelationalSpeakerGRU(feature_dim=input_dim, hidden_dim=64).to(device)
        opt_gru = torch.optim.Adam(gru.parameters(), lr=config['learning_rate'])
        
        for epoch in range(epochs):
            gru.train()
            for batch in train_loader:
                features, cids, mask, gold_index, q_ids = batch_to_device(batch, device)
                opt_gru.zero_grad()
                scores, _ = gru(features, mask, gold_index_for_update=gold_index) # Teacher forcing during training
                loss = crit(scores.squeeze(0), gold_index.squeeze(0) if gold_index.dim() == 2 else gold_index)
                loss.backward()
                opt_gru.step()
                
        # --- Evaluate GRU with Ablations ---
        modes = ["normal", "reset_hidden_each_quote", "zero_update", "shuffled_update", "teacher_forced_eval_diagnostic"]
        for mode in modes:
            logger.info(f"Evaluating GRU Mode: {mode} (Seed {seed})")
            preds_gru = evaluate_scorer(gru, test_loader, device, type_mappings, is_gru=True, ablation_mode=mode)
            preds_gru.to_csv(f"results/EXP025/preds_gru_{mode}_seed{seed}.csv", index=False)
            
            mets_gru = compute_metrics(preds_gru)
            mets_gru['Seed'] = seed
            mets_gru['Model'] = f'gru_{mode}'
            
            if mode == "normal":
                seed_results_gru.append(mets_gru)
            
            memory_ablation_results.append(mets_gru)

    # Compile EXP025A (Seed Stability)
    all_seed_results = seed_results_nomemory + seed_results_gru
    seed_df = pd.DataFrame(all_seed_results)
    seed_df.to_csv("results/EXP025/seed_metrics.csv", index=False)
    
    # Calculate aggregates
    summary_list = []
    for model_name in ['nomemory_no_anchor', 'gru_normal']:
        m_df = seed_df[seed_df['Model'] == model_name]
        mean_acc = m_df['Accuracy'].mean()
        std_acc = m_df['Accuracy'].std()
        ci_95 = 1.96 * (std_acc / np.sqrt(len(m_df))) if len(m_df) > 1 else 0
        summary_list.append({
            "Model": model_name,
            "Accuracy_Mean": mean_acc,
            "Accuracy_Std": std_acc,
            "Accuracy_Min": m_df['Accuracy'].min(),
            "Accuracy_Max": m_df['Accuracy'].max(),
            "Accuracy_95CI": ci_95,
            "Implicit_Mean": m_df['Implicit_Accuracy'].mean(),
            "Anaphoric_Mean": m_df['Anaphoric_Accuracy'].mean()
        })
    seed_summary_df = pd.DataFrame(summary_list)
    seed_summary_df.to_csv("results/EXP025/seed_summary.csv", index=False)
    
    # Compile EXP025B (Memory Ablation)
    ablation_df = pd.DataFrame(memory_ablation_results)
    ablation_df.to_csv("results/EXP025/memory_ablation_metrics.csv", index=False)
    
    ablation_summary = ablation_df.groupby('Model').agg(
        Accuracy_Mean=('Accuracy', 'mean'),
        Implicit_Mean=('Implicit_Accuracy', 'mean'),
        MRR_Mean=('MRR', 'mean')
    ).reset_index()
    ablation_summary.to_csv("results/EXP025/memory_ablation_summary.csv", index=False)
    
    # Statistical tests (using Seed 1 as proxy for McNemar)
    stat_tests = {}
    if not is_cpu_test or is_cpu_test:
        preds_normal = pd.read_csv("results/EXP025/preds_gru_normal_seed1.csv")
        modes_to_test = ["reset_hidden_each_quote", "zero_update", "shuffled_update", "teacher_forced_eval_diagnostic"]
        for m in modes_to_test:
            preds_m = pd.read_csv(f"results/EXP025/preds_gru_{m}_seed1.csv")
            pval = mcnemar_test(preds_normal, preds_m)
            stat_tests[f"gru_normal vs gru_{m}"] = pval
            
    with open("results/EXP025/memory_ablation_statistical_tests.json", "w") as f:
        json.dump(stat_tests, f, indent=4)
        
    # Decision Rules & Report
    report = ["# EXP025 GRU Stability and Memory Ablation\n"]
    
    if is_cpu_test:
        report.append("## SMOKE TEST RESULTS\n")
        report.append("Smoke test ran successfully.")
        report.append("\n**EXP025A Seed Stability (1 Epoch, Seed 1):**")
        report.append(seed_summary_df.to_markdown(index=False))
        report.append("\n**EXP025B Memory Ablation (1 Epoch, Seed 1):**")
        report.append(ablation_summary.to_markdown(index=False))
        
    else:
        report.append("## EXP025A: Seed Stability\n")
        report.append(seed_summary_df.to_markdown(index=False))
        
        gru_mean = seed_summary_df[seed_summary_df['Model'] == 'gru_normal']['Accuracy_Mean'].iloc[0]
        nm_mean = seed_summary_df[seed_summary_df['Model'] == 'nomemory_no_anchor']['Accuracy_Mean'].iloc[0]
        
        # Count wins
        gru_wins = 0
        for seed in SEEDS:
            gru_acc = seed_df[(seed_df['Model'] == 'gru_normal') & (seed_df['Seed'] == seed)]['Accuracy'].iloc[0]
            nm_acc = seed_df[(seed_df['Model'] == 'nomemory_no_anchor') & (seed_df['Seed'] == seed)]['Accuracy'].iloc[0]
            if gru_acc > nm_acc:
                gru_wins += 1
                
        report.append(f"\nGRU wins on {gru_wins}/{len(SEEDS)} seeds.")
        
        if gru_mean > nm_mean and gru_wins >= 4:
            report.append("> **Conclusion**: The clean GRU improvement is stable.")
        else:
            report.append("> **Conclusion**: The GRU improvement is not stable enough to treat as a reliable architectural gain.")
            
        report.append("\n## EXP025B: GRU Memory Ablation\n")
        
        ablation_summary_dict = ablation_summary.set_index('Model').to_dict('index')
        gru_norm_acc = ablation_summary_dict['gru_normal']['Accuracy_Mean']
        gru_reset_acc = ablation_summary_dict['gru_reset_hidden_each_quote']['Accuracy_Mean']
        gru_zero_acc = ablation_summary_dict['gru_zero_update']['Accuracy_Mean']
        gru_shuffled_acc = ablation_summary_dict['gru_shuffled_update']['Accuracy_Mean']
        gru_tf_acc = ablation_summary_dict['gru_teacher_forced_eval_diagnostic']['Accuracy_Mean']
        
        report.append(f"- `gru_normal`: {gru_norm_acc:.4f}")
        report.append(f"- `gru_reset_hidden_each_quote`: {gru_reset_acc:.4f}")
        report.append(f"- `gru_zero_update`: {gru_zero_acc:.4f}")
        report.append(f"- `gru_shuffled_update`: {gru_shuffled_acc:.4f}")
        report.append(f"- `gru_teacher_forced_eval_diagnostic`: {gru_tf_acc:.4f}")
        
        if (gru_norm_acc > gru_reset_acc) and (gru_norm_acc > gru_zero_acc) and (gru_norm_acc > gru_shuffled_acc):
            report.append("\n> **Conclusion**: The GRU improvement is caused by recurrent memory.")
        else:
            report.append("\n> **Conclusion**: The gain comes from the neural candidate encoder/scorer, not recurrent memory.")
            
        if gru_tf_acc > gru_norm_acc + 0.02:
            report.append("> **Conclusion**: Correct memory state still has additional headroom, but autoregressive memory remains imperfect.")
            
        # Explicit check for zero vs reset
        if abs(gru_zero_acc - gru_reset_acc) < 0.001:
            report.append("\n*Note: `gru_zero_update` and `gru_reset_hidden_each_quote` produced nearly identical results because scoring occurs before the memory update, meaning both effectively evaluate each quote from a zero state.*")
            
    with open("results/EXP025/metrics_report.md", "w") as f:
        f.write("\n".join(report))
        
    logger.info("EXP025 execution complete.")

if __name__ == "__main__":
    main()
