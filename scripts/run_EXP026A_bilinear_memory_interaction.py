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
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.device import get_device
from src.evaluation.runner import load_frozen_exp014_dataset
from src.neural.sequence_dataset import build_character_vocab, TensorSequenceDataset
from src.neural.models import (
    EXP026ABilinearSpeakerGRU,
    EXP026ACandidateOnlyScorer,
    EXP026AParameterMatchedNoMemoryScorer,
    RelationalSpeakerGRU,
)
from scripts.run_EXP021A_2_mlp_ce import compute_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STOPPING_RULE = (
    "Regardless of statistical significance, EXP026A is the final scorer-integration "
    "experiment for the current feature-only GRU architecture."
)

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

ABLATION_MODES = [
    "normal",
    "zero_state",
    "shuffled_update",
    "teacher_forced_eval_diagnostic",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
        batch.novel_id,
    )


def collate_one(batch):
    return batch[0]


def reverse_vocab(vocab):
    return {idx: name for name, idx in vocab.items()}


def display_candidate(novel_id, candidate_id, rev_vocab):
    raw = rev_vocab.get(int(candidate_id), "<UNK>")
    prefix = f"{novel_id}::"
    return raw[len(prefix):] if raw.startswith(prefix) else raw


def load_quote_types(test_df):
    q_info_dir = Path("data/raw/pdnc/data")
    type_mappings = {}
    for novel in test_df['novel'].unique():
        q_info_path = q_info_dir / novel / "quotation_info.csv"
        if not q_info_path.exists():
            continue
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


def run_model_scores(model, features, mask, gold_index, variant, mode):
    if variant.startswith("nomemory"):
        return model(features, mask)

    if gold_index.dim() == 1:
        gold_index = gold_index.unsqueeze(0)

    if mode == "teacher_forced_eval_diagnostic":
        return model(features, mask, gold_index_for_update=gold_index)
    if mode == "zero_state":
        return model(features, mask, gold_index_for_update=None, ablate_memory=True)
    if mode == "shuffled_update":
        torch.manual_seed(42)
        return model(features, mask, gold_index_for_update=None, ablate_shuffle=True)
    return model(features, mask, gold_index_for_update=None)


def evaluate_scorer(model, dataloader, device, type_mappings, vocab, variant, mode):
    model.eval()
    rows = []
    rev_vocab = reverse_vocab(vocab)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for batch in dataloader:
            features, cids, mask, gold_index, q_ids, novel_id = batch_to_device(batch, device)
            scores, diagnostics = run_model_scores(model, features, mask, gold_index, variant, mode)
            scores = scores.squeeze(0)
            mask_2d = mask.squeeze(0)
            cids_2d = cids.squeeze(0)
            gold_index_1d = gold_index.squeeze(0) if gold_index.dim() == 2 else gold_index
            losses = loss_fn(scores, gold_index_1d).detach().cpu().numpy()
            probs = torch.softmax(scores, dim=-1)
            sorted_indices = torch.argsort(scores, dim=-1, descending=True)

            interaction_diag = None
            if diagnostics is not None and variant == "gru_bilinear_interaction":
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
                    "loss": losses[i],
                    "gold_probability": probs[i, gold].item(),
                    "evaluation_mode": mode,
                    "interaction_scores": json.dumps(interaction_values),
                })

    return pd.DataFrame(rows)


def train_model(model, train_loader, device, lr, epochs, variant):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            features, _, mask, gold_index, _, _ = batch_to_device(batch, device)
            optimizer.zero_grad()

            if variant.startswith("nomemory"):
                scores, _ = model(features, mask)
            else:
                scores, _ = model(features, mask, gold_index_for_update=gold_index)

            targets = gold_index.squeeze(0) if gold_index.dim() == 2 else gold_index
            loss = criterion(scores.squeeze(0), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info("%s epoch %d/%d loss %.4f", variant, epoch + 1, epochs, total_loss / len(train_loader))


def trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def parameter_report(models):
    rows = []
    for name, model in models.items():
        scorer_params = 0
        if hasattr(model, "candidate_score_branch"):
            scorer_params += trainable_params(model.candidate_score_branch)
        if hasattr(model, "scorer"):
            scorer_params += trainable_params(model.scorer)
        if hasattr(model, "bilinear_weight"):
            scorer_params += model.bilinear_weight.numel()

        rows.append({
            "Model": name,
            "Trainable_Params": trainable_params(model),
            "Scorer_Only_Params": scorer_params,
            "Bilinear_W_Params": getattr(model, "bilinear_weight", torch.empty(0)).numel(),
        })
    return pd.DataFrame(rows)


def make_parameter_matched_control(input_dim, target_params, base_hidden_dim=64):
    best_model = None
    best_gap = None
    for scorer_hidden_dim in range(16, 513):
        model = EXP026AParameterMatchedNoMemoryScorer(
            input_dim,
            hidden_dim=base_hidden_dim,
            scorer_hidden_dim=scorer_hidden_dim,
        )
        gap = abs(trainable_params(model) - target_params)
        if best_gap is None or gap < best_gap:
            best_model = model
            best_gap = gap
    return best_model


def summarize_seed_metrics(metrics_rows):
    seed_df = pd.DataFrame(metrics_rows)
    summary = seed_df.groupby("Model").agg(
        Accuracy_Mean=("Accuracy", "mean"),
        Accuracy_Std=("Accuracy", "std"),
        Accuracy_Min=("Accuracy", "min"),
        Accuracy_Max=("Accuracy", "max"),
        Implicit_Mean=("Implicit_Accuracy", "mean"),
        Anaphoric_Mean=("Anaphoric_Accuracy", "mean"),
        MRR_Mean=("MRR", "mean"),
    ).reset_index()
    return seed_df, summary


def prediction_level_ablation(normal_df, ablated_df, model_name, ablation_mode):
    merged = normal_df.merge(
        ablated_df,
        on="quote_id",
        suffixes=("_normal", "_ablated"),
    )
    normal_correct = merged["correct_normal"].astype(bool)
    ablated_correct = merged["correct_ablated"].astype(bool)
    agreement = (
        merged["predicted_candidate_index_normal"] == merged["predicted_candidate_index_ablated"]
    ).mean()
    recoveries = ((~normal_correct) & ablated_correct).sum()
    regressions = (normal_correct & (~ablated_correct)).sum()
    gold_prob_change = (
        merged["gold_probability_normal"] - merged["gold_probability_ablated"]
    ).mean()
    divergences = []
    for _, row in merged.iterrows():
        p = np.array(json.loads(row["candidate_probabilities_normal"]), dtype=float)
        q = np.array(json.loads(row["candidate_probabilities_ablated"]), dtype=float)
        eps = 1e-12
        divergences.append(float(np.sum(p * np.log((p + eps) / (q + eps)))))

    base = {
        "Model": model_name,
        "Ablation": ablation_mode,
        "Top1_Agreement": agreement,
        "Recoveries": int(recoveries),
        "Regressions": int(regressions),
        "Mean_Gold_Probability_Change": gold_prob_change,
        "Mean_Distribution_KL": float(np.mean(divergences)) if divergences else 0.0,
    }
    rows = [base]

    for quote_type, q_df in merged.groupby("quote_type_normal"):
        type_row = dict(base)
        type_row["Quote_Type"] = quote_type
        type_row["Top1_Agreement"] = (
            q_df["predicted_candidate_index_normal"] == q_df["predicted_candidate_index_ablated"]
        ).mean()
        type_row["Recoveries"] = int(((~q_df["correct_normal"].astype(bool)) & q_df["correct_ablated"].astype(bool)).sum())
        type_row["Regressions"] = int((q_df["correct_normal"].astype(bool) & (~q_df["correct_ablated"].astype(bool))).sum())
        type_row["Mean_Gold_Probability_Change"] = (
            q_df["gold_probability_normal"] - q_df["gold_probability_ablated"]
        ).mean()
        rows.append(type_row)

    return rows


def interaction_magnitude_report(preds_df):
    values = []
    for raw in preds_df["interaction_scores"]:
        if pd.isna(raw) or raw == "[]":
            continue
        values.extend(abs(float(v)) for v in json.loads(raw))

    if not values:
        return {
            "Mean_Abs_Interaction": 0.0,
            "Median_Abs_Interaction": 0.0,
            "P95_Abs_Interaction": 0.0,
        }
    arr = np.array(values)
    return {
        "Mean_Abs_Interaction": float(np.mean(arr)),
        "Median_Abs_Interaction": float(np.median(arr)),
        "P95_Abs_Interaction": float(np.percentile(arr, 95)),
    }


def previous_speaker_labels(df, seq_dataset):
    labels = {}
    for seq in seq_dataset:
        novel_df = df[df["novel"] == seq.novel_id].copy()
        for q_id in seq.quote_ids:
            quote_df = novel_df[novel_df["quote_id"] == q_id]
            if "candidate_is_previous_speaker" not in quote_df.columns:
                continue
            labels[(seq.novel_id, q_id)] = quote_df["candidate_is_previous_speaker"].astype(int).tolist()
    return labels


def collect_probe_rows(model, seq_dataset, labels_by_quote, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for seq in seq_dataset:
            features = seq.candidate_features.to(device)
            mask = seq.candidate_mask.to(device)
            gold_index = seq.gold_index.to(device)
            h = torch.zeros(1, model.hidden_dim, device=device)

            for t, q_id in enumerate(seq.quote_ids):
                cand_vecs = model.encode_candidates(features[t])
                labels = labels_by_quote.get((seq.novel_id, q_id), [])
                for c in range(mask[t].sum().item()):
                    rows.append({
                        "novel": seq.novel_id,
                        "quote_id": q_id,
                        "quote_type": None,
                        "candidate_repr": cand_vecs[c].detach().cpu().numpy(),
                        "hidden_repr": h.squeeze(0).detach().cpu().numpy(),
                        "label": int(labels[c]) if c < len(labels) else 0,
                    })
                spk_vec = cand_vecs[gold_index[t].item()].unsqueeze(0)
                h = model.gru_cell(spk_vec, h)
    return rows


def probe_matrix(rows, mode):
    candidate = np.stack([r["candidate_repr"] for r in rows])
    hidden = np.stack([r["hidden_repr"] for r in rows])
    if mode == "candidate_only":
        return candidate
    return np.concatenate([candidate, hidden], axis=1)


def run_probe_controls(model, train_seq, test_seq, train_df, test_df, device, seed):
    train_labels = previous_speaker_labels(train_df, train_seq)
    test_labels = previous_speaker_labels(test_df, test_seq)
    train_rows = collect_probe_rows(model, train_seq, train_labels, device)
    test_rows = collect_probe_rows(model, test_seq, test_labels, device)

    y_train = np.array([r["label"] for r in train_rows])
    y_test = np.array([r["label"] for r in test_rows])
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return pd.DataFrame([{
            "Seed": seed,
            "Probe": "skipped_single_class",
            "Previous_Speaker_PR_AUC": np.nan,
        }])

    probe_rows = []
    rng = np.random.default_rng(seed)
    shuffled_test_rows = [dict(r) for r in test_rows]
    for novel in sorted({r["novel"] for r in shuffled_test_rows}):
        idxs = [i for i, r in enumerate(shuffled_test_rows) if r["novel"] == novel]
        hidden_values = [shuffled_test_rows[i]["hidden_repr"] for i in idxs]
        rng.shuffle(hidden_values)
        for i, hidden in zip(idxs, hidden_values):
            shuffled_test_rows[i]["hidden_repr"] = hidden

    for probe_name, train_mode, test_source in [
        ("candidate_only", "candidate_only", test_rows),
        ("candidate_plus_aligned_hidden", "candidate_plus_hidden", test_rows),
        ("candidate_plus_within_novel_shuffled_hidden", "candidate_plus_hidden", shuffled_test_rows),
    ]:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        clf.fit(probe_matrix(train_rows, train_mode), y_train)
        probs = clf.predict_proba(probe_matrix(test_source, train_mode))[:, 1]
        probe_rows.append({
            "Seed": seed,
            "Probe": probe_name,
            "Previous_Speaker_PR_AUC": average_precision_score(y_test, probs),
        })

    return pd.DataFrame(probe_rows)


def write_report(out_dir, seed_df, summary_df, ablation_df, interaction_df, param_df, probe_df):
    report = ["# EXP026A Bilinear Candidate-History Compatibility\n"]
    report.append(STOPPING_RULE)
    report.append("\n## Design\n")
    report.append("- Variant A: `nomemory_candidate_only`, candidate encoder plus `f(c)`, no GRU.")
    report.append("- Variant B: `gru_concatenative_current`, current EXP025 GRU scorer.")
    report.append("- Variant C: `gru_bilinear_interaction`, `s(c,h)=f(c)+c^T W h`.")
    report.append("- No auxiliary loss, no candidate-feature dropout, no alternate fusion scorer.")
    report.append("\n## Parameter Counts\n")
    report.append(param_df.to_markdown(index=False))
    report.append("\n\nBilinear interaction adds `64 x 64 = 4096` parameters. A parameter-matched no-memory control is required only if Variant C exceeds Variant A by more than 10% trainable parameters.")
    report.append("\n## Seed Summary\n")
    report.append(summary_df.to_markdown(index=False))
    report.append("\n## Ablation Summary\n")
    report.append(ablation_df.to_markdown(index=False))
    report.append("\n## Interaction Magnitude\n")
    report.append(interaction_df.to_markdown(index=False))
    report.append("\n## Probe Controls\n")
    report.append(probe_df.to_markdown(index=False))
    report.append("\n## Acceptance Rules\n")
    report.append("- Predictive: Variant C must exceed the five-seed no-memory baseline 71.97% by at least +0.5 pp, with positive paired seed delta and wins on most seeds.")
    report.append("- Causal: normal - zero_state must be at least +0.5 pp, normal must exceed shuffled_update, and teacher-forced must not be materially or consistently worse than normal.")
    report.append("- Probe success alone cannot override failed attribution results.")
    report.append("- If `c^T W h` is near zero almost everywhere, do not interpret any accuracy gain as memory interaction.")
    report.append("\n## Outcome Rules\n")
    report.append("- Accuracy and memory dependence both succeed: promote Variant C, then interpret what memory learned.")
    report.append("- Accuracy improves but zero-state does not hurt: treat the gain as static ranking, not recurrent memory.")
    report.append("- Memory sensitivity increases but accuracy does not: EXP026B auxiliary supervision is justified.")
    report.append("- Neither improves: retire the current feature-only GRU branch; do not run EXP026B, EXP026C, deeper fusion, attention, or another scorer-integration variant.")

    with open(out_dir / "EXP026A_BILINEAR_MEMORY_INTERACTION_REPORT.md", "w") as f:
        f.write("\n".join(report))


def main():
    out_dir = Path("results/EXP026A")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/EXP021_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = get_device() if config['device'] == 'auto' else config['device']
    logger.info("Using device: %s", device)

    df = load_frozen_exp014_dataset()
    vocab = build_character_vocab(df, vocab_path=str(out_dir / "character_vocab.json"))

    missing = sorted(set(APPROVED_STATE_FREE_FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Missing approved features: {missing}")

    present_forbidden = sorted(set(APPROVED_STATE_FREE_FEATURES) & FORBIDDEN_FEATURES)
    if present_forbidden:
        raise ValueError(f"Forbidden features present in approved features: {present_forbidden}")

    with open(out_dir / "feature_list.json", "w") as f:
        json.dump({
            "approved_features": APPROVED_STATE_FREE_FEATURES,
            "forbidden_features_absent": True,
            "stopping_rule": STOPPING_RULE,
        }, f, indent=4)

    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()

    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    train_df_scaled[APPROVED_STATE_FREE_FEATURES] = scaler.fit_transform(train_df[APPROVED_STATE_FREE_FEATURES])
    test_df_scaled[APPROVED_STATE_FREE_FEATURES] = scaler.transform(test_df[APPROVED_STATE_FREE_FEATURES])

    train_seq = TensorSequenceDataset(train_df_scaled, APPROVED_STATE_FREE_FEATURES, feature_mode='all', vocab=vocab, scaler=None)
    test_seq = TensorSequenceDataset(test_df_scaled, APPROVED_STATE_FREE_FEATURES, feature_mode='all', vocab=vocab, scaler=None)
    train_loader = DataLoader(train_seq, batch_size=1, shuffle=True, collate_fn=collate_one)
    test_loader = DataLoader(test_seq, batch_size=1, shuffle=False, collate_fn=collate_one)

    type_mappings = load_quote_types(test_df)
    input_dim = len(APPROVED_STATE_FREE_FEATURES)
    is_cpu_test = os.environ.get("CPU_TEST_RUN") == "1"
    seeds = [1] if is_cpu_test else [1, 2, 3, 4, 5]
    epochs = 1 if is_cpu_test else config['epochs']

    run_config = {
        "experiment_name": "EXP026A_bilinear_memory_interaction",
        "stopping_rule": STOPPING_RULE,
        "seeds": seeds,
        "epochs": epochs,
        "learning_rate": config["learning_rate"],
        "input_dim": input_dim,
        "hidden_dim": 64,
        "cpu_test_run": is_cpu_test,
        "variants": [
            "nomemory_candidate_only",
            "gru_concatenative_current",
            "gru_bilinear_interaction",
        ],
        "primary_variants": [
            "nomemory_candidate_only",
            "gru_concatenative_current",
            "gru_bilinear_interaction",
        ],
        "parameter_matched_control_rule": (
            "Add nomemory_parameter_matched only if gru_bilinear_interaction has "
            "more than 10% additional trainable parameters relative to nomemory_candidate_only."
        ),
        "ablation_modes": ABLATION_MODES,
        "feature_list": APPROVED_STATE_FREE_FEATURES,
        "forbidden_feature_list": sorted(FORBIDDEN_FEATURES),
        "no_auxiliary_loss": True,
        "no_candidate_feature_dropout": True,
        "no_alternate_interaction_scorers": True,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=4)

    seed_metrics = []
    ablation_metrics = []
    ablation_detail_rows = []
    interaction_rows = []
    probe_frames = []
    all_param_frames = []

    for seed in seeds:
        logger.info("========== SEED %d ==========", seed)
        set_seed(seed)
        models = {
            "nomemory_candidate_only": EXP026ACandidateOnlyScorer(input_dim, hidden_dim=64).to(device),
            "gru_concatenative_current": RelationalSpeakerGRU(input_dim, hidden_dim=64).to(device),
            "gru_bilinear_interaction": EXP026ABilinearSpeakerGRU(input_dim, hidden_dim=64).to(device),
        }

        variant_a_params = trainable_params(models["nomemory_candidate_only"])
        variant_c_params = trainable_params(models["gru_bilinear_interaction"])
        if variant_c_params > variant_a_params * 1.10:
            matched = make_parameter_matched_control(input_dim, variant_c_params, base_hidden_dim=64).to(device)
            models["nomemory_parameter_matched"] = matched

        for model in models.values():
            model.feature_names = APPROVED_STATE_FREE_FEATURES

        param_df = parameter_report(models)
        param_df["Seed"] = seed
        all_param_frames.append(param_df)

        for variant, model in models.items():
            set_seed(seed)
            logger.info("Training %s seed %d", variant, seed)
            train_model(model, train_loader, device, config["learning_rate"], epochs, variant)

            modes = ["normal"] if variant.startswith("nomemory") else ABLATION_MODES
            mode_predictions = {}
            for mode in modes:
                logger.info("Evaluating %s %s seed %d", variant, mode, seed)
                preds = evaluate_scorer(model, test_loader, device, type_mappings, vocab, variant, mode)
                preds["seed"] = seed
                preds.to_csv(out_dir / f"preds_{variant}_{mode}_seed{seed}.csv", index=False)
                mode_predictions[mode] = preds

                metrics = compute_metrics(preds)
                metrics["Seed"] = seed
                metrics["Model"] = variant if mode == "normal" else f"{variant}_{mode}"
                metrics["Evaluation_Mode"] = mode
                if mode == "normal":
                    seed_metrics.append(metrics)
                else:
                    ablation_metrics.append(metrics)

                if variant == "gru_bilinear_interaction" and mode == "normal":
                    row = interaction_magnitude_report(preds)
                    row["Seed"] = seed
                    row["Model"] = variant
                    interaction_rows.append(row)

            if variant in {"gru_concatenative_current", "gru_bilinear_interaction"}:
                normal_df = mode_predictions["normal"]
                for mode in ["zero_state", "shuffled_update", "teacher_forced_eval_diagnostic"]:
                    ablation_detail_rows.extend(
                        prediction_level_ablation(normal_df, mode_predictions[mode], variant, mode)
                    )

            if variant == "gru_bilinear_interaction":
                probe_frames.append(
                    run_probe_controls(model, train_seq, test_seq, train_df_scaled, test_df_scaled, device, seed)
                )

    seed_df, seed_summary = summarize_seed_metrics(seed_metrics)
    seed_df.to_csv(out_dir / "seed_metrics.csv", index=False)
    seed_summary.to_csv(out_dir / "seed_summary.csv", index=False)

    ablation_df = pd.DataFrame(ablation_metrics)
    ablation_df.to_csv(out_dir / "memory_ablation_metrics.csv", index=False)
    ablation_summary = ablation_df.groupby("Model").agg(
        Accuracy_Mean=("Accuracy", "mean"),
        Implicit_Mean=("Implicit_Accuracy", "mean"),
        Anaphoric_Mean=("Anaphoric_Accuracy", "mean"),
        MRR_Mean=("MRR", "mean"),
    ).reset_index()
    ablation_summary.to_csv(out_dir / "memory_ablation_summary.csv", index=False)

    pd.DataFrame(ablation_detail_rows).to_csv(out_dir / "prediction_level_ablation_metrics.csv", index=False)

    interaction_df = pd.DataFrame(interaction_rows)
    interaction_df.to_csv(out_dir / "interaction_magnitude_metrics.csv", index=False)

    param_report = pd.concat(all_param_frames, ignore_index=True)
    param_report.to_csv(out_dir / "parameter_counts.csv", index=False)

    probe_df = pd.concat(probe_frames, ignore_index=True) if probe_frames else pd.DataFrame()
    probe_df.to_csv(out_dir / "previous_speaker_probe_controls.csv", index=False)

    write_report(out_dir, seed_df, seed_summary, ablation_summary, interaction_df, param_report, probe_df)
    logger.info("EXP026A implementation run complete.")


if __name__ == "__main__":
    main()
