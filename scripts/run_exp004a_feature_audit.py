import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir

setup_logging()
logger = get_logger("exp004a_feature_audit")

def run_feature_audit():
    logger.info("Starting EXP004A: Feature Audit...")
    
    input_file = get_data_dir() / "phase2" / "candidate_features.csv"
    if not input_file.exists():
        logger.error(f"Dataset not found at {input_file}. Run generate_dataset_p2.py first.")
        return
        
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} candidate rows.")
    
    # Baseline Statistics
    total_quotes = df['quote_id'].nunique()
    logger.info(f"Total Unique Quotes: {total_quotes}")
    
    label_counts = df['label'].value_counts(normalize=True)
    logger.info(f"Class Imbalance: {label_counts.to_dict()}")
    
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    logger.info(f"Train/Test split: {len(train_df)} / {len(test_df)} candidates")
    
    # Feature Audit
    features = [c for c in df.columns if c not in ["quote_id", "novel", "candidate", "gold_speaker", "split", "label"]]
    
    audit_data = []
    for f in features:
        col = df[f]
        missing_pct = col.isnull().mean() * 100
        unique_vals = col.nunique()
        mean_val = col.mean() if pd.api.types.is_numeric_dtype(col) else np.nan
        var_val = col.var() if pd.api.types.is_numeric_dtype(col) else np.nan
        min_val = col.min() if pd.api.types.is_numeric_dtype(col) else np.nan
        max_val = col.max() if pd.api.types.is_numeric_dtype(col) else np.nan
        
        # Correlation with label
        corr = col.corr(df['label']) if pd.api.types.is_numeric_dtype(col) else np.nan
        
        audit_data.append({
            "Feature": f,
            "Type": str(col.dtype),
            "Mean": mean_val,
            "Variance": var_val,
            "Min": min_val,
            "Max": max_val,
            "Missing %": missing_pct,
            "Unique Values": unique_vals,
            "Correlation with Label": corr
        })
        
    audit_df = pd.DataFrame(audit_data)
    
    EXP_DIR = get_reports_dir() / "EXP004A"
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    output_csv = EXP_DIR / "feature_summary.csv"
    audit_df.to_csv(output_csv, index=False)
    
    logger.info(f"Feature Audit saved to {output_csv}")
    
    # Basic sanity checks
    zero_variance = audit_df[audit_df['Variance'] == 0.0]['Feature'].tolist()
    if zero_variance:
        logger.warning(f"WARNING: Found features with zero variance: {zero_variance}")
        
    high_missing = audit_df[audit_df['Missing %'] > 0]['Feature'].tolist()
    if high_missing:
        logger.warning(f"WARNING: Found features with missing values: {high_missing}")

if __name__ == "__main__":
    run_feature_audit()
