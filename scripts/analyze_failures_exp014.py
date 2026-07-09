import os
import pandas as pd
from pathlib import Path
import random

def get_data_dir() -> Path:
    return Path("data")

def analyze_exp014_residuals():
    # Load the EXP013 oracle analysis which contains the baseline_rank (EXP012B predictions)
    oracle_file = Path("results/EXP013/prediction_comparison.csv")
    if not oracle_file.exists():
        print(f"Error: {oracle_file} not found.")
        return
        
    oracle_df = pd.read_csv(oracle_file)
    
    # We want EXP012B wrong predictions
    failures_df = oracle_df[oracle_df['baseline_rank'] > 1].copy()
    
    # Load candidate_features to get byte spans
    features_df = pd.read_csv("data/raw/pdnc/phase2/candidate_features.csv")
    features_df = features_df.groupby('quote_id').first().reset_index()
    
    failures_df = failures_df.merge(features_df[['quote_id', 'quote_start_byte', 'quote_end_byte']], on='quote_id', how='left')
    
    print(f"Total EXP012B failures found: {len(failures_df)}")
    
    # Randomly sample ~50 failures
    sample_size = min(50, len(failures_df))
    sampled = failures_df.sample(sample_size, random_state=42)
    
    os.makedirs("results/EXP014A", exist_ok=True)
    out_path = "results/EXP014A/raw_failure_sample.md"
    
    with open(out_path, "w", encoding='utf-8') as out:
        out.write("# EXP014A: Residual Error Sample (EXP012B Failures)\n\n")
        out.write(f"Randomly sampled {sample_size} failures out of {len(failures_df)}.\n\n")
        
        for _, row in sampled.iterrows():
            novel = row['novel']
            novel_txt_path = Path(f"data/raw/pdnc/data/{novel}/{novel}.txt")
            if not novel_txt_path.exists():
                txt_files = list(Path(f"data/raw/pdnc/data/{novel}").glob("*.txt"))
                if txt_files:
                    novel_txt_path = txt_files[0]
                    
            text = "TEXT NOT FOUND"
            if novel_txt_path.exists():
                with open(novel_txt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    start = int(row['quote_start_byte'])
                    end = int(row['quote_end_byte'])
                    ctx_start = max(0, start - 200)
                    ctx_end = min(len(content), end + 200)
                    text = f"...{content[ctx_start:start]}**{content[start:end]}**{content[end:ctx_end]}..."
            
            # Clean up newlines for markdown
            text = text.replace('\n', ' ')
            
            out.write(f"### Quote ID: {row['quote_id']} ({row['novel']})\n")
            out.write(f"- **Gold Speaker**: {row['gold_candidate']}\n")
            out.write(f"- **Context**: {text.strip()}\n\n")
            
    print(f"Sample generated at {out_path}")

if __name__ == "__main__":
    analyze_exp014_residuals()
