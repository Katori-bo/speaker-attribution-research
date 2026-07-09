import os
import json
import time
import tracemalloc
import pandas as pd
import logging
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, log_loss
from sklearn.inspection import permutation_importance

from src.coreference.parser import BookNLPParser
from src.coreference.mapping import MentionToEntityMapper
from src.attribution.pipeline import AttributionFeatureProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    "features": {
        "coreference": True,
        "attribution": True
    }
}

def get_data_dir() -> Path:
    return Path("data")

def get_novel_text(novel: str) -> str:
    novel_txt_path = Path(f"data/raw/pdnc/data/{novel}/{novel}.txt")
    if not novel_txt_path.exists():
        txt_files = list(Path(f"data/raw/pdnc/data/{novel}").glob("*.txt"))
        if txt_files:
            novel_txt_path = txt_files[0]
    with open(novel_txt_path, 'r', encoding='utf-8') as f:
        return f.read()

def run_exp014():
    logger.info("Starting EXP014F: Attribution Role Feature Evaluation")
    
    exp012_cache_file = get_data_dir() / "raw" / "pdnc" / "phase2" / "candidate_features_exp012.csv"
    if not exp012_cache_file.exists():
        logger.error(f"Cannot find EXP012 cached features at {exp012_cache_file}.")
        return
        
    df = pd.read_csv(exp012_cache_file)
    
    
    novel_features_list = []
    
    start_time = time.time()
    tracemalloc.start()
    
    for novel, novel_df in df.groupby('novel'):
        logger.info(f"Processing novel: {novel}")
        content = get_novel_text(novel)
        
        novel_dir = os.path.join("data/raw/pdnc/booknlp_out", novel)
        entities_path = os.path.join(novel_dir, f"{novel}.entities")
        book_path = os.path.join(novel_dir, f"{novel}.book")
        
        parser = BookNLPParser()
        entities = parser.parse_entities(entities_path)
        aliases = parser.parse_book_aliases(book_path)
        mapper = MentionToEntityMapper(entities, aliases)
        
        attr_provider = AttributionFeatureProvider(mapper, enabled=CONFIG["features"]["attribution"])
        
        unique_quotes = novel_df['quote_id'].unique()
        def get_q_idx(q_id):
            try: return int(q_id.split('_')[-1])
            except: return 0
        unique_quotes = sorted(unique_quotes, key=get_q_idx)
        
        for q_id in unique_quotes:
            q_df = novel_df[novel_df['quote_id'] == q_id]
            q_start = int(q_df['quote_start_byte'].iloc[0])
            q_end = int(q_df['quote_end_byte'].iloc[0])
            
            for _, row in q_df.iterrows():
                candidate = row['candidate']
                candidate_chain_id = mapper.resolve_string_to_chain_id(candidate)
                if candidate_chain_id is None:
                    candidate_chain_id = -1
                
                # Attribution features
                attr_feats = attr_provider.get_features(
                    candidate_chain_id=int(candidate_chain_id),
                    quote_id=q_id,
                    quote_start=q_start,
                    quote_end=q_end,
                    content=content
                )
                
                attr_feats['quote_id'] = q_id
                attr_feats['candidate'] = candidate
                novel_features_list.append(attr_feats)
                
    attr_feat_df = pd.DataFrame(novel_features_list)
    df = df.merge(attr_feat_df, on=['quote_id', 'candidate'], how='left')
    
    extraction_time = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    base_feats = [c for c in df.columns if c not in [
        "quote_id", "novel", "candidate", "gold_speaker", "split", "label",
        "quote_start_byte", "quote_end_byte", "quoteByteSpans",
        "candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency",
        "candidate_is_attributed_speaker"
    ] and not c.startswith("symbolic_")]
    
    if CONFIG["features"]["coreference"]:
        base_feats += ["candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency"]
        
    exp_feats = base_feats.copy()
    if CONFIG["features"]["attribution"]:
        exp_feats += ["candidate_is_attributed_speaker"]
        
    # Baseline Model (EXP012B)
    baseline_model = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    baseline_model.fit(train_df[base_feats], train_df['label'])
    test_df['baseline_score'] = baseline_model.predict_proba(test_df[base_feats])[:, 1]
    
    # Experimental Model (EXP014)
    exp_model = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    exp_model.fit(train_df[exp_feats], train_df['label'])
    test_df['exp_score'] = exp_model.predict_proba(test_df[exp_feats])[:, 1]
    
    # Ranks
    test_df['baseline_rank'] = test_df.groupby('quote_id')['baseline_score'].rank(ascending=False, method='first')
    test_df['exp_rank'] = test_df.groupby('quote_id')['exp_score'].rank(ascending=False, method='first')
    
    # Metrics
    def compute_metrics(y_true, y_score, groups, ranks):
        top_preds = ranks == 1
        return {
            "Accuracy": y_true[top_preds].mean(),
            "Precision": precision_score(y_true, (y_score > 0.5).astype(int)),
            "Recall": recall_score(y_true, (y_score > 0.5).astype(int)),
            "F1": f1_score(y_true, (y_score > 0.5).astype(int)),
            "PR-AUC": average_precision_score(y_true, y_score),
            "LogLoss": log_loss(y_true, y_score)
        }
    
    baseline_metrics = compute_metrics(test_df['label'].values, test_df['baseline_score'].values, test_df['quote_id'].values, test_df['baseline_rank'].values)
    exp_metrics = compute_metrics(test_df['label'].values, test_df['exp_score'].values, test_df['quote_id'].values, test_df['exp_rank'].values)
    
    os.makedirs("results/EXP014", exist_ok=True)
    
    with open("results/EXP014/metrics.json", "w") as f:
        json.dump({
            "Baseline": baseline_metrics,
            "Experimental": exp_metrics,
            "Runtime_Seconds": extraction_time,
            "Peak_Memory_MB": peak_mem / 10**6
        }, f, indent=4)
        
    # Permutation Importance
    logger.info("Computing Permutation Importance...")
    pi = permutation_importance(exp_model, test_df[exp_feats], test_df['label'], n_repeats=5, random_state=42, scoring='roc_auc')
    importances = {exp_feats[i]: pi.importances_mean[i] for i in range(len(exp_feats))}
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    with open("results/EXP014/feature_importance.json", "w") as f:
        json.dump(sorted_importances, f, indent=4)
        
    # Oracle / Recovery Analysis
    detailed_results = []
    a_correct = b_recovered = c_regression = d_wrong = 0
    exp012_errors = 0
    
    activated_cases = 0
    recovered_errors = 0
    new_regressions = 0
    exp012_correct_and_activated = 0
    exp014_correct_and_activated = 0
    confidence_movements = []
    
    for q_id, q_df in test_df.groupby('quote_id'):
        gold_row = q_df[q_df['label'] == 1]
        if len(gold_row) == 0: continue
        gold_row = gold_row.iloc[0]
        
        b_rank = gold_row['baseline_rank']
        e_rank = gold_row['exp_rank']
        feature_active = (q_df['candidate_is_attributed_speaker'] == 1.0).any()
        gold_active = (gold_row['candidate_is_attributed_speaker'] == 1.0)
        
        if b_rank > 1: exp012_errors += 1
            
        if b_rank == 1 and e_rank == 1:
            cat = "A. Both correct"
            a_correct += 1
        elif b_rank > 1 and e_rank == 1:
            cat = "B. Newly recovered"
            b_recovered += 1
        elif b_rank == 1 and e_rank > 1:
            cat = "C. Regression"
            c_regression += 1
        else:
            cat = "D. Both wrong"
            d_wrong += 1
            
        if feature_active:
            activated_cases += 1
            if b_rank == 1:
                exp012_correct_and_activated += 1
            if e_rank == 1:
                exp014_correct_and_activated += 1
            if b_rank > 1 and e_rank == 1:
                recovered_errors += 1
            if b_rank == 1 and e_rank > 1:
                new_regressions += 1
                
            confidence_movements.append({
                "quote_id": q_id,
                "novel": gold_row['novel'],
                "gold_candidate": gold_row['candidate'],
                "baseline_prob": gold_row['baseline_score'],
                "exp_prob": gold_row['exp_score'],
                "gold_active": gold_active,
                "category": cat
            })
            
        detailed_results.append({
            "quote_id": q_id,
            "novel": gold_row['novel'],
            "gold_candidate": gold_row['candidate'],
            "baseline_rank": b_rank,
            "exp_rank": e_rank,
            "category": cat,
            "attribution_active": feature_active
        })
        
    pred_df = pd.DataFrame(detailed_results)
    pred_df.to_csv("results/EXP014/prediction_comparison.csv", index=False)
    
    # Write activation analysis
    with open("results/EXP014/activation_analysis.md", "w") as f:
        f.write("# EXP014 Activation Analysis\n\n")
        f.write(f"**Activated cases (attribution feature == 1 for some candidate):** {activated_cases}\n")
        f.write(f"**EXP012 correct:** {exp012_correct_and_activated}\n")
        f.write(f"**EXP014 correct:** {exp014_correct_and_activated}\n")
        f.write(f"**Recovered errors:** {recovered_errors}\n")
        f.write(f"**New regressions:** {new_regressions}\n\n")
        
        # Confidence Movement
        if confidence_movements:
            conf_df = pd.DataFrame(confidence_movements)
            mean_b_prob = conf_df['baseline_prob'].mean()
            mean_e_prob = conf_df['exp_prob'].mean()
            f.write("## Confidence Movement (on Activated Quotes)\n")
            f.write(f"Mean EXP012 gold probability: {mean_b_prob:.4f}\n")
            f.write(f"Mean EXP014 gold probability: {mean_e_prob:.4f}\n")
            f.write(f"Average probability gain: {mean_e_prob - mean_b_prob:.4f}\n")
            
    # Write regression audit
    with open("results/EXP014/regression_audit.md", "w") as f:
        f.write("# Regression Audit (Attribution Active)\n\n")
        reg_df = pd.DataFrame(confidence_movements)
        if len(reg_df) > 0:
            regressions = reg_df[reg_df['category'] == 'C. Regression']
            f.write(f"Found {len(regressions)} regressions where attribution was active.\n\n")
            for _, r in regressions.iterrows():
                f.write(f"- Quote: {r['quote_id']} | Novel: {r['novel']} | Gold: {r['gold_candidate']} | Gold Active? {r['gold_active']}\n")
        else:
            f.write("Found 0 regressions where attribution was active.\n")
            
    # Write overall results
    with open("results/EXP014/EXP014_RESULTS.md", "w") as f:
        f.write("# EXP014 Evaluation Results\n\n")
        f.write("## Metrics\n")
        f.write(f"- Accuracy (Baseline): {baseline_metrics['Accuracy']:.4f}\n")
        f.write(f"- Accuracy (EXP014): {exp_metrics['Accuracy']:.4f}\n")
        f.write(f"- Absolute Accuracy Gain: {exp_metrics['Accuracy'] - baseline_metrics['Accuracy']:.4f}\n")
        f.write(f"- PR-AUC (Baseline): {baseline_metrics['PR-AUC']:.4f}\n")
        f.write(f"- PR-AUC (EXP014): {exp_metrics['PR-AUC']:.4f}\n")
        f.write(f"- LogLoss (Baseline): {baseline_metrics['LogLoss']:.4f}\n")
        f.write(f"- LogLoss (EXP014): {exp_metrics['LogLoss']:.4f}\n\n")
        
        f.write("## Feature Importance Rank\n")
        f.write("candidate_is_attributed_speaker importance: ")
        rank = list(sorted_importances.keys()).index("candidate_is_attributed_speaker") + 1
        total_feats = len(sorted_importances)
        val = sorted_importances['candidate_is_attributed_speaker']
        f.write(f"Rank {rank} / {total_feats} (Score: {val:.4f})\n\n")

if __name__ == "__main__":
    run_exp014()
