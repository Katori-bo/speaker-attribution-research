import os
import pandas as pd
import logging
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features.conversation_extractor import ConversationFeatureExtractor
from src.discourse.conversation_state import ConversationStateModule
from src.coreference.pipeline import SemanticFeatureProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_data_dir() -> Path:
    return Path("data")

def get_ranking_metrics(y_true, y_score, groups):
    df = pd.DataFrame({'label': y_true, 'score': y_score, 'group': groups})
    df['rank'] = df.groupby('group')['score'].rank(ascending=False, method='first')
    
    # Accuracy is proportion of groups where the top ranked candidate is the true speaker
    top_preds = df[df['rank'] == 1]
    accuracy = top_preds['label'].mean()
    
    # Mean Reciprocal Rank
    mrr_sum = 0
    total_groups = len(df['group'].unique())
    for name, group in df.groupby('group'):
        gold_ranks = group[group['label'] == 1]['rank'].values
        if len(gold_ranks) > 0:
            mrr_sum += 1.0 / gold_ranks[0]
            
    # Mean Rank
    mean_rank_sum = 0
    total = 0
    for name, group in df.groupby('group'):
        gold_ranks = group[group['label'] == 1]['rank'].values
        if len(gold_ranks) > 0:
            mean_rank_sum += gold_ranks[0]
            total += 1
            
    return {
        "Accuracy": accuracy,
        "MRR": mrr_sum / total_groups if total_groups > 0 else 0,
        "Mean_Rank": mean_rank_sum / total if total > 0 else 0,
    }

def run_exp012():
    logger.info("Starting EXP012: Semantic Representation Evaluation (Generalization)")
    
    exp012_cache_file = get_data_dir() / "raw" / "pdnc" / "phase2" / "candidate_features_exp012.csv"
    if exp012_cache_file.exists():
        logger.info(f"Loading cached EXP012 features from {exp012_cache_file}")
        df = pd.read_csv(exp012_cache_file)
    else:
        input_file = get_data_dir() / "raw" / "pdnc" / "phase2" / "candidate_features.csv"
        if not input_file.exists():
            input_file = get_data_dir() / "phase2" / "candidate_features.csv"
            
        df = pd.read_csv(input_file)
        
        # 2. Add EXP011 Discourse State Features (Baseline for EXP012)
        logger.info(f"Extracting EXP011 Discourse State Features for {len(df['quote_id'].unique())} quotes...")
        extractor = ConversationFeatureExtractor()
        novel_features_list = []
        
        for novel, novel_df in df.groupby('novel'):
            state = ConversationStateModule(novel)
            unique_quotes = novel_df['quote_id'].unique()
            for q_id in unique_quotes:
                q_df = novel_df[novel_df['quote_id'] == q_id]
                quote_dict = {
                    "quote_start_byte": q_df.iloc[0].get("quote_start_byte", -1),
                    "quote_end_byte": q_df.iloc[0].get("quote_end_byte", -1)
                }
                
                for _, row in q_df.iterrows():
                    candidate = row['candidate']
                    f = extractor.extract(quote_dict, candidate, state)
                    f['quote_id'] = q_id
                    f['candidate'] = candidate
                    novel_features_list.append(f)
                    
                gold_rows = q_df[q_df['label'] == 1]
                if len(gold_rows) > 0:
                    gold_speaker = gold_rows.iloc[0]['candidate']
                else:
                    gold_speaker = "Unknown"
                    
                if q_df.iloc[0]['discourse_dialogue_position'] == 1:
                    state.reset(novel)
                    
                state.update(quote_dict, gold_speaker)

        exp011_feat_df = pd.DataFrame(novel_features_list)
        df = df.merge(exp011_feat_df, on=['quote_id', 'candidate'], how='left')
        
        # 3. Add EXP012 Semantic Features
        logger.info("Extracting EXP012 Semantic Features (Full Dataset)...")
        semantic_provider = SemanticFeatureProvider()
        df = semantic_provider.augment_features(df)
        
        # Save to cache
        logger.info(f"Saving EXP012 augmented dataset to {exp012_cache_file}")
        df.to_csv(exp012_cache_file, index=False)
        
    # 4. Train/Eval Splits
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    base_feats = [c for c in df.columns if c not in [
        "quote_id", "novel", "candidate", "gold_speaker", "split", "label",
        "quote_start_byte", "quote_end_byte", "quoteByteSpans",
        "candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency"
    ] and not c.startswith("symbolic_")]
    
    exp_feats = base_feats + [
        "candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency"
    ]
    
    logger.info(f"Baseline Features (EXP011): {base_feats}")
    logger.info(f"Experimental Features (EXP012): {exp_feats}")
    
    # Baseline Model
    baseline_model = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    baseline_model.fit(train_df[base_feats], train_df['label'])
    test_df['baseline_score'] = baseline_model.predict_proba(test_df[base_feats])[:, 1]
    
    # Experimental Model
    exp_model = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    exp_model.fit(train_df[exp_feats], train_df['label'])
    test_df['exp_score'] = exp_model.predict_proba(test_df[exp_feats])[:, 1]

    # Overall Evaluation
    baseline_metrics = get_ranking_metrics(test_df['label'], test_df['baseline_score'], test_df['quote_id'])
    exp_metrics = get_ranking_metrics(test_df['label'], test_df['exp_score'], test_df['quote_id'])
    
    logger.info(f"--- OVERALL RESULTS ---")
    logger.info(f"Baseline Accuracy (EXP011): {baseline_metrics['Accuracy']:.4f}")
    logger.info(f"Experimental Accuracy (EXP012): {exp_metrics['Accuracy']:.4f}")
    logger.info(f"Delta: {exp_metrics['Accuracy'] - baseline_metrics['Accuracy']:.4f}")
    
    # Calculate Recovery & Oracle Analysis
    test_df['baseline_rank'] = test_df.groupby('quote_id')['baseline_score'].rank(ascending=False, method='first')
    test_df['exp_rank'] = test_df.groupby('quote_id')['exp_score'].rank(ascending=False, method='first')
    
    # Mapping Oracle Analysis
    logger.info("--- MAPPING ORACLE ANALYSIS ---")
    type_a_failures = 0
    type_b_failures = 0
    recovered_quotes = 0
    new_error_quotes = 0
    
    # We will identify failure types for the gold candidates in quotes where EXP012 failed
    novel_results = []
    detailed_oracle_results = []
    
    for novel, ndf in test_df.groupby('novel'):
        n_quotes = len(ndf['quote_id'].unique())
        if n_quotes == 0:
            continue
            
        b_metrics = get_ranking_metrics(ndf['label'], ndf['baseline_score'], ndf['quote_id'])
        e_metrics = get_ranking_metrics(ndf['label'], ndf['exp_score'], ndf['quote_id'])
        
        novel_results.append({
            "Novel": novel,
            "Quotes": n_quotes,
            "Baseline Acc": b_metrics['Accuracy'],
            "Exp Acc": e_metrics['Accuracy'],
            "Delta": e_metrics['Accuracy'] - b_metrics['Accuracy']
        })
        
        # Oracle analysis per quote in novel
        gold_cands = ndf[ndf['label'] == 1]
        for _, row in gold_cands.iterrows():
            q_id = row['quote_id']
            b_rank = row['baseline_rank']
            e_rank = row['exp_rank']
            
            # Determine mapping success
            # If nearest_coref_dist is exactly -1 (the missing value), it was not mapped or not found.
            # A more robust check is asking the provider directly, but we can infer from features:
            # If all coref features are missing, mapping failed or entity has no coref chain.
            features_missing = (row['nearest_coref_dist'] == -1 and row['chain_recency'] == -1)
            
            if b_rank > 1 and e_rank == 1:
                recovered_quotes += 1
                cat = "RECOVERED"
            elif b_rank == 1 and e_rank > 1:
                new_error_quotes += 1
                cat = "REGRESSION"
            elif b_rank > 1 and e_rank > 1:
                cat = "UNCHANGED_FAILURE"
            else:
                cat = "UNCHANGED_SUCCESS"
                
            type_flag = ""
            if e_rank > 1:
                # Experimental model failed to rank gold speaker first
                if features_missing:
                    type_b_failures += 1
                    type_flag = "TYPE_B"
                else:
                    type_a_failures += 1
                    type_flag = "TYPE_A"
                    
            detailed_oracle_results.append({
                "quote_id": q_id,
                "novel": novel,
                "gold_candidate": row['candidate'],
                "baseline_rank": b_rank,
                "exp_rank": e_rank,
                "category": cat,
                "failure_type": type_flag
            })

    logger.info(f"Total EXP012 Errors: {type_a_failures + type_b_failures}")
    logger.info(f"Type A (Mapped successfully, but prediction failed): {type_a_failures}")
    logger.info(f"Type B (Mapping failed or coref missing): {type_b_failures}")
    logger.info(f"Recovered: {recovered_quotes}, New Errors: {new_error_quotes}")

    logger.info("\n--- PER-NOVEL BREAKDOWN ---")
    novel_df = pd.DataFrame(novel_results).sort_values(by="Delta", ascending=False)
    # Rename columns to match requirements
    novel_df = novel_df.rename(columns={
        "Novel": "novel", 
        "Baseline Acc": "baseline_accuracy", 
        "Exp Acc": "semantic_accuracy", 
        "Delta": "delta"
    })
    logger.info("\n" + novel_df.to_markdown(index=False))
    
    # Save results
    os.makedirs("results/EXP012B", exist_ok=True)
    
    # evaluation.csv
    eval_df = novel_df.copy()
    eval_df.to_csv("results/EXP012B/evaluation.csv", index=False)
    
    # prediction_changes.csv (we'll just output the summary metrics as requested, 
    # but the instructions asked for a csv with RECOVERED, REGRESSION, UNCHANGED_SUCCESS, UNCHANGED_FAILURE)
    unchanged_success = 0
    unchanged_failure = 0
    
    # Calculate unchanged counts based on ranks
    gold_cands = test_df[test_df['label'] == 1]
    for _, row in gold_cands.iterrows():
        b_rank = row['baseline_rank']
        e_rank = row['exp_rank']
        if b_rank == 1 and e_rank == 1:
            unchanged_success += 1
        elif b_rank > 1 and e_rank > 1:
            unchanged_failure += 1
            
    changes_df = pd.DataFrame([{
        "RECOVERED": recovered_quotes,
        "REGRESSION": new_error_quotes,
        "UNCHANGED_SUCCESS": unchanged_success,
        "UNCHANGED_FAILURE": unchanged_failure,
        "TYPE_A_FAILURES": type_a_failures,
        "TYPE_B_FAILURES": type_b_failures
    }])
    changes_df.to_csv("results/EXP012B/prediction_changes.csv", index=False)
    
    # Save detailed oracle analysis
    detailed_oracle_df = pd.DataFrame(detailed_oracle_results)
    detailed_oracle_df.to_csv("results/EXP012B/oracle_analysis.csv", index=False)
    
if __name__ == "__main__":
    run_exp012()
