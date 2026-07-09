import pandas as pd
import numpy as np
import csv
import re
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir
from src.models.classical_models import PointwiseLogisticRanker

setup_logging()
logger = get_logger("exp010_semantic_taxonomy")

def get_raw_quote(quote_id):
    parts = quote_id.split('_')
    novel_name = parts[0]
    try:
        idx = int(parts[1])
    except:
        return {}
        
    csv_file = get_data_dir() / "data" / novel_name / "quotation_info.csv"
    if not csv_file.exists(): return {}
        
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
        if idx < len(reader): return reader[idx]
    return {}

def extract_evidence(raw_data, group_df):
    ref_expr = raw_data.get('referringExpression', '')
    quote_text = raw_data.get('quoteText', '')
    
    # Pronouns
    pronouns = re.findall(r'\b(he|she|him|her|his|hers|they|them|their|the former|the latter)\b', ref_expr, re.IGNORECASE)
    
    # Proper Names (Capitalized words that aren't the start of the string, simple proxy)
    # Actually, just finding any capitalized words in ref_expr
    proper_names = re.findall(r'\b[A-Z][a-z]+\b', ref_expr)
    
    # Dialogue tags
    tags = re.findall(r'\b(said|replied|asked|exclaimed|cried|whispered|shouted|added|continued|answered|remarked)\b', ref_expr, re.IGNORECASE)
    
    gold_row = group_df[group_df['label'] == 1].iloc[0]
    has_explicit = (gold_row['candidate_is_recent_mention'] == 1 or 
                    gold_row['candidate_is_previous_speaker'] == 1 or 
                    gold_row['candidate_is_explicit_mention'] == 1)
    
    prev_speaker = "Unknown"
    for _, row in group_df.iterrows():
        if row['candidate_is_previous_speaker'] == 1:
            prev_speaker = row['candidate']
            break
            
    return {
        "Quote": quote_text,
        "Context (Referring Expression)": ref_expr,
        "Explicit Signals Present?": "Yes" if has_explicit else "No",
        "Pronouns Detected": ", ".join(set([p.lower() for p in pronouns])) if pronouns else "None",
        "Proper Names Detected": ", ".join(set(proper_names)) if proper_names else "None",
        "Dialogue Tag Detected": ", ".join(set([t.lower() for t in tags])) if tags else "None",
        "Previous Speaker": prev_speaker,
        "Conversation Turn Index": gold_row['conversation_turn_index'],
        "Discourse Context Length": gold_row['discourse_context_length']
    }

def run_exp010():
    logger.info("Starting EXP010A: Generating Annotation Worksheet...")
    
    input_file = get_data_dir() / "phase2" / "candidate_features.csv"
    df = pd.read_csv(input_file)
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    all_features = [c for c in df.columns if c not in ["quote_id", "novel", "candidate", "gold_speaker", "split", "label"] and not c.startswith("symbolic_")]
    top_3 = ['candidate_is_explicit_mention', 'candidate_is_previous_speaker', 'candidate_is_recent_mention']
    
    lr = PointwiseLogisticRanker(random_state=42)
    lr.fit(train_df[top_3], train_df['label'])
    test_df['lr_score'] = lr.predict_proba(test_df[top_3])
    
    gbm = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    gbm.fit(train_df[all_features], train_df['label'])
    test_df['gbm_score'] = gbm.predict_proba(test_df[all_features])[:, 1]
    
    quotes = []
    
    for quote_id, group in test_df.groupby('quote_id'):
        if group['label'].sum() > 0:
            lr_correct = group.loc[group['lr_score'].idxmax(), 'label'] == 1
            gbm_correct = group.loc[group['gbm_score'].idxmax(), 'label'] == 1
            
            gold_row = group[group['label'] == 1].iloc[0]
            has_explicit = (gold_row['candidate_is_recent_mention'] == 1 or 
                            gold_row['candidate_is_previous_speaker'] == 1 or 
                            gold_row['candidate_is_explicit_mention'] == 1)
            
            category = None
            if not lr_correct and not gbm_correct:
                if has_explicit:
                    category = "Explicit_Present_But_Fail"
                else:
                    category = "Both_Fail"
            elif not lr_correct and gbm_correct:
                category = "GBM_Only"
                
            if category:
                raw_data = get_raw_quote(quote_id)
                evidence = extract_evidence(raw_data, group)
                
                lr_pred_candidate = group.loc[group['lr_score'].idxmax(), 'candidate']
                gbm_pred_candidate = group.loc[group['gbm_score'].idxmax(), 'candidate']
                
                row_data = {
                    "Quote_ID": quote_id,
                    "Failure_Category": category,
                    "Gold_Speaker": gold_row['gold_speaker'],
                    "Candidate_List": " | ".join(group['candidate'].tolist()),
                    **evidence,
                    "ANNOTATION: Primary Category": "",
                    "ANNOTATION: Secondary Category": "",
                    "ANNOTATION: Evidence": "",
                    "ANNOTATION: Context Window Needed": "",
                    "ANNOTATION: Confidence": "",
                    "ANNOTATION: Explicit Alternative Feasible?": "",
                    "ANNOTATION: Notes": "",
                    "LR_Prediction": lr_pred_candidate,
                    "GBM_Prediction": gbm_pred_candidate
                }
                quotes.append(row_data)
                
    q_df = pd.DataFrame(quotes)
    
    # Stratified Sampling
    np.random.seed(42)
    sample_both_fail = q_df[q_df['Failure_Category'] == 'Both_Fail'].sample(n=min(100, len(q_df[q_df['Failure_Category'] == 'Both_Fail'])))
    sample_gbm_only = q_df[q_df['Failure_Category'] == 'GBM_Only'].sample(n=min(50, len(q_df[q_df['Failure_Category'] == 'GBM_Only'])))
    sample_explicit_fail = q_df[q_df['Failure_Category'] == 'Explicit_Present_But_Fail'].sample(n=min(50, len(q_df[q_df['Failure_Category'] == 'Explicit_Present_But_Fail'])))
    
    final_sample = pd.concat([sample_both_fail, sample_gbm_only, sample_explicit_fail]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    EXP_DIR = get_reports_dir() / "EXP010"
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_file = EXP_DIR / "annotation_worksheet.csv"
    final_sample.to_csv(csv_file, index=False)
    
    logger.info(f"Worksheet generated at {csv_file} with {len(final_sample)} samples.")
    logger.info("Fill out the 'ANNOTATION: ...' columns manually for EXP010B.")

if __name__ == "__main__":
    run_exp010()
