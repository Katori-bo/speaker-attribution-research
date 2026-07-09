import pandas as pd
import json
import numpy as np
import os
from pathlib import Path
from src.utils.logger import setup_logging, get_logger

logger = get_logger("EXP018A_Analysis")

def build_quote_type_map(novels):
    qt_map = {}
    q_info_dir = "data/raw/pdnc/data"
    for novel in novels:
        q_info = pd.read_csv(os.path.join(q_info_dir, novel, "quotation_info.csv"))
        for _, row in q_info.iterrows():
            q_id = row.get("quote_id")
            if not q_id: q_id = f"{novel}_{row.name}"
            q_type = str(row['quoteType'])
            
            if q_type == 'Implicit':
                final_type = 'Implicit'
            elif q_type == 'Anaphoric':
                final_type = 'Anaphoric'
            elif q_type == 'Explicit':
                final_type = 'Explicit'
            else:
                final_type = 'Implicit'
                
            qt_map[q_id] = final_type
    return qt_map

def compute_accuracy(df):
    """Computes basic accuracy from predictions."""
    if 'score' in df.columns:
        idx = df.groupby('quote_id', sort=False)['score'].idxmax()
        chosen = df.loc[idx]
    else:
        chosen = df
        
    correct = (chosen['candidate'] == chosen['gold_speaker']).sum()
    total = len(chosen)
    return correct / total if total > 0 else 0.0

def compute_implicit_accuracy(df, qt_map):
    """Computes accuracy on implicit quotes only."""
    if 'score' in df.columns:
        idx = df.groupby('quote_id', sort=False)['score'].idxmax()
        chosen = df.loc[idx].copy()
    else:
        chosen = df.copy()
        
    chosen['quote_type'] = chosen['quote_id'].map(qt_map)
    implicit = chosen[chosen['quote_type'] == 'Implicit']
    correct = (implicit['candidate'] == implicit['gold_speaker']).sum()
    total = len(implicit)
    return correct / total if total > 0 else 0.0

def main():
    setup_logging()
    logger.info("Starting EXP018A Analysis...")
    
    k_values = [1, 3, 5, 10, 20]
    results = []
    
    # Load quote type map first
    # Just need it from the K=1 dataframe to get the novels
    res_dir_k1 = Path(f"results/EXP018A/beam_K1")
    if res_dir_k1.exists():
        df_k1 = pd.read_csv(res_dir_k1 / "predictions.csv")
        novels = df_k1['novel'].unique()
        qt_map = build_quote_type_map(novels)
    else:
        qt_map = {}
    
    for k in k_values:
        res_dir = Path(f"results/EXP018A/beam_K{k}")
        pred_file = res_dir / "predictions.csv"
        oracle_file = res_dir / "oracle_survival.json"
        
        if not pred_file.exists() or not oracle_file.exists():
            logger.warning(f"Results for K={k} not found. Skipping.")
            continue
            
        df = pd.read_csv(pred_file)
        with open(oracle_file, "r") as f:
            oracle_stats = json.load(f)
            
        acc = compute_accuracy(df)
        imp_acc = compute_implicit_accuracy(df, qt_map)
        
        total_quotes = oracle_stats.get("total_quotes", 1)
        gold_survived = oracle_stats.get("gold_path_survived", 0)
        oracle_survival = gold_survived / total_quotes if total_quotes > 0 else 0.0
        
        results.append({
            "K": k,
            "Accuracy": acc,
            "Implicit_Accuracy": imp_acc,
            "Oracle_Survival": oracle_survival
        })
        
    if not results:
        logger.error("No results found to analyze.")
        return
        
    results_df = pd.DataFrame(results)
    print("\n--- EXP018A Beam Search Sweep Results ---")
    print(results_df.to_string(index=False))
    
    # Save to CSV
    out_dir = Path("results/EXP018A")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "beam_sweep.csv", index=False)
    logger.info(f"Saved analysis to {out_dir}/beam_sweep.csv")

if __name__ == "__main__":
    main()
