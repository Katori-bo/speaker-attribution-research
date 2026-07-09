import pandas as pd
import numpy as np
from pathlib import Path
import json

from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir
from src.models.classical_models import PointwiseLogisticRanker
from sklearn.ensemble import HistGradientBoostingClassifier

from src.discourse.conversation_state import ConversationStateModule
from src.features.conversation_extractor import ConversationFeatureExtractor

setup_logging()
logger = get_logger("exp011a_representation")

def get_ranking_metrics(y_true, y_prob, group_ids):
    df = pd.DataFrame({'group': group_ids, 'y_true': y_true, 'score': y_prob})
    
    mrr_sum = 0
    mean_rank_sum = 0
    recall_1 = 0
    total = 0
    
    for _, g in df.groupby('group'):
        if g['y_true'].sum() > 0:
            g = g.sort_values(by='score', ascending=False).reset_index(drop=True)
            gold_idx = g[g['y_true'] == 1].index[0]
            rank = gold_idx + 1
            
            mrr_sum += 1.0 / rank
            mean_rank_sum += rank
            if rank == 1:
                recall_1 += 1
            total += 1
            
    return {
        "Accuracy": recall_1 / total if total > 0 else 0,
        "MRR": mrr_sum / total if total > 0 else 0,
        "Mean_Rank": mean_rank_sum / total if total > 0 else 0,
    }

def run_exp011a():
    logger.info("Starting EXP011A: Conversation State Representation Evaluation")
    
    # 1. Load the frozen feature dataset (PDNC)
    input_file = get_data_dir() / "raw" / "pdnc" / "phase2" / "candidate_features.csv"
    if not input_file.exists():
        input_file = get_data_dir() / "phase2" / "candidate_features.csv"
    
    df = pd.read_csv(input_file)
    
    # 2. Extract Novel Features by Re-Simulating the Novel State Module
    # The dataset is ordered by quote_id (e.g. AHandfulOfDust_0, _1, _2...)
    # We will simulate the conversation state per novel.
    logger.info("Simulating ConversationStateModule...")
    
    extractor = ConversationFeatureExtractor()
    novel_features_list = []
    
    for novel, novel_df in df.groupby('novel'):
        state = ConversationStateModule(novel)
        
        # Iterate over quotes in sequence
        unique_quotes = novel_df['quote_id'].unique()
        
        for q_id in unique_quotes:
            q_df = novel_df[novel_df['quote_id'] == q_id]
            
            # Extract features for all candidates for this quote
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
                
            # Update state using the GOLD speaker to simulate perfect tracking,
            gold_rows = q_df[q_df['label'] == 1]
            if len(gold_rows) > 0:
                gold_speaker = gold_rows.iloc[0]['candidate']
            else:
                gold_speaker = "Unknown"
                
            # Detect naive scene boundary if discourse_dialogue_position resets
            if q_df.iloc[0]['discourse_dialogue_position'] == 1:
                state.reset(novel)
                
            state.update(quote_dict, gold_speaker)

    new_feat_df = pd.DataFrame(novel_features_list)
    df = df.merge(new_feat_df, on=['quote_id', 'candidate'], how='left')
    
    # 3. Train/Eval Baseline (Frozen EXP009 Features)
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    baseline_features = [c for c in df.columns if c not in [
        "quote_id", "novel", "candidate", "gold_speaker", "split", "label",
        "quote_start_byte", "quote_end_byte",
        "conv_active_id", "conv_interruption_distance", 
        "candidate_in_participant_stack", "candidate_stack_depth"
    ] and not c.startswith("symbolic_")]
    
    logger.info(f"Baseline Features: {baseline_features}")
    
    # Fit frozen pipeline on baseline features
    baseline_model = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    baseline_model.fit(train_df[baseline_features], train_df['label'])
    baseline_probs = baseline_model.predict_proba(test_df[baseline_features])[:, 1]
    test_df['baseline_score'] = baseline_probs
    
    baseline_metrics = get_ranking_metrics(test_df['label'], test_df['baseline_score'], test_df['quote_id'])
    
    # 4. Train/Eval Experimental (Baseline + Novel Features)
    experimental_features = baseline_features + [
        "conv_active_id", "conv_interruption_distance", 
        "candidate_in_participant_stack", "candidate_stack_depth"
    ]
    
    logger.info(f"Experimental Features: {experimental_features}")
    
    exp_model = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    exp_model.fit(train_df[experimental_features], train_df['label'])
    exp_probs = exp_model.predict_proba(test_df[experimental_features])[:, 1]
    test_df['exp_score'] = exp_probs
    
    exp_metrics = get_ranking_metrics(test_df['label'], test_df['exp_score'], test_df['quote_id'])
    
    # 5. Calculate Net Gain / New Errors
    test_df['baseline_pred'] = test_df.groupby('quote_id')['baseline_score'].transform(lambda x: x == x.max())
    test_df['exp_pred'] = test_df.groupby('quote_id')['exp_score'].transform(lambda x: x == x.max())
    
    quotes_eval = test_df[test_df['label'] == 1].copy()
    
    baseline_correct = quotes_eval['baseline_pred'] == True
    exp_correct = quotes_eval['exp_pred'] == True
    
    new_errors = quotes_eval[baseline_correct & ~exp_correct]
    recovered = quotes_eval[~baseline_correct & exp_correct]
    
    # 6. Cross-reference with EXP010 Annotations (Secondary Metrics)
    exp010_file = Path("results") / "EXP010" / "semantic_annotations_master.csv"
    if exp010_file.exists():
        exp010_df = pd.read_csv(exp010_file)
        if 'Quote_ID' in exp010_df.columns:
            exp010_df = exp010_df.rename(columns={'Quote_ID': 'quote_id'})
        
        # Determine subset recovery
        # Merge quotes_eval with exp010_df
        subset_eval = quotes_eval.merge(exp010_df, on='quote_id', how='inner')
        
        spk_cont = subset_eval[subset_eval['ANNOTATION: Primary Category'] == 'Discourse: Speaker Continuity']
        coref = subset_eval[subset_eval['ANNOTATION: Primary Category'] == 'Reference: Pronominal Coreference']
        
        spk_cont_rec = len(spk_cont[~spk_cont['baseline_pred'] & spk_cont['exp_pred']])
        coref_rec = len(coref[~coref['baseline_pred'] & coref['exp_pred']])
        
        logger.info(f"Speaker Continuity Recovered: {spk_cont_rec} / {len(spk_cont)}")
        logger.info(f"Coreference Recovered: {coref_rec} / {len(coref)}")
    else:
        logger.warning(f"Could not find EXP010 annotations at {exp010_file}")

    # Output Results
    logger.info(f"Baseline Accuracy: {baseline_metrics['Accuracy']:.4f}")
    logger.info(f"Experimental Accuracy: {exp_metrics['Accuracy']:.4f}")
    logger.info(f"Recovered: {len(recovered)}, New Errors: {len(new_errors)}")
    
    # Save Report
    out_dir = Path("results") / "EXP011"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "STATUS.md", "w") as f:
        f.write("# EXP011A Status\n\n")
        f.write(f"- Baseline Accuracy: {baseline_metrics['Accuracy']:.4f}\n")
        f.write(f"- Experimental Accuracy: {exp_metrics['Accuracy']:.4f}\n")
        f.write(f"- Net Gain: {len(recovered) - len(new_errors)} quotes\n")
        f.write(f"- Recovered (Total): {len(recovered)}\n")
        f.write(f"- New Errors Introduced: {len(new_errors)}\n")
        
if __name__ == "__main__":
    run_exp011a()
