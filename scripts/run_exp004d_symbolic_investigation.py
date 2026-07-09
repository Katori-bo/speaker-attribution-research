import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir

setup_logging()
logger = get_logger("exp004d_symbolic_investigation")

def run_symbolic_investigation():
    logger.info("Starting EXP004D: Symbolic Feature Investigation...")
    
    input_file = get_data_dir() / "phase2" / "candidate_features.csv"
    if not input_file.exists():
        logger.error("Dataset not found.")
        return
        
    df = pd.read_csv(input_file)
    train_df = df[df['split'] == 'train']
    
    # Hypotheses to Test:
    # H1 & H2: Multicollinearity / Redundancy
    # Measure correlation between Symbolic rules and Candidate features
    
    candidate_features = [c for c in df.columns if c.startswith("candidate_")]
    symbolic_features = [c for c in df.columns if c.startswith("symbolic_")]
    
    logger.info("Calculating Feature Correlations...")
    corr_matrix = train_df[candidate_features + symbolic_features].corr()
    
    # Extract only the symbolic vs candidate correlations
    sym_vs_cand_corr = corr_matrix.loc[symbolic_features, candidate_features]
    
    EXP_DIR = get_reports_dir() / "EXP004D"
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(EXP_DIR / "symbolic_investigation_report.md", 'w') as f:
        f.write("# EXP004D: Symbolic Feature Investigation Report\n\n")
        f.write("## Hypothesis 1 & 2: Multicollinearity and Redundancy\n")
        f.write("Do symbolic features perfectly duplicate continuous candidate features?\n\n")
        
        f.write("| Symbolic Feature | Candidate Feature | Pearson Correlation |\n")
        f.write("|------------------|-------------------|---------------------|\n")
        
        for sym_feat in symbolic_features:
            for cand_feat in candidate_features:
                c = sym_vs_cand_corr.loc[sym_feat, cand_feat]
                if abs(c) > 0.3: # Only show meaningful correlations
                    f.write(f"| {sym_feat} | {cand_feat} | {c:.4f} |\n")
                    
        f.write("\n## Hypothesis 3: Noisy Rules\n")
        f.write("Are the symbolic rules introducing noise (i.e. firing confidently on incorrect candidates)?\n\n")
        
        f.write("| Symbolic Feature | Fired Count | Precision (Correct when fired) |\n")
        f.write("|------------------|-------------|--------------------------------|\n")
        
        for sym_feat in symbolic_features:
            fired_mask = train_df[sym_feat] == 1.0
            fired_count = fired_mask.sum()
            if fired_count > 0:
                correct_count = (train_df[fired_mask]['label'] == 1).sum()
                precision = correct_count / fired_count
                f.write(f"| {sym_feat} | {fired_count} | {precision*100:.2f}% |\n")
                
        f.write("\n## Conclusion\n")
        f.write("Based on the data above, we can draw a conclusion about why the linear model performed better without symbolic features.\n")
        
    logger.info(f"Investigation report saved to {EXP_DIR / 'symbolic_investigation_report.md'}")

if __name__ == "__main__":
    run_symbolic_investigation()
