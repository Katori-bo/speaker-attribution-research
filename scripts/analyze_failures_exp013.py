import pandas as pd
from pathlib import Path
import random

def analyze():
    # Load detailed oracle failures
    oracle_file = Path("results/EXP012B/oracle_analysis.csv")
    if not oracle_file.exists():
        print(f"Error: {oracle_file} not found. Please run run_exp012.py first.")
        return
        
    oracle_df = pd.read_csv(oracle_file)
    
    # Filter to the categories we care about
    target_df = oracle_df[oracle_df['category'].isin(['UNCHANGED_FAILURE', 'REGRESSION'])]
    
    # Load candidate_features.csv to get the byte spans
    features_df = pd.read_csv("data/raw/pdnc/phase2/candidate_features.csv")
    features_df = features_df.groupby('quote_id').first().reset_index()
    
    target_df = target_df.merge(features_df[['quote_id', 'quote_start_byte', 'quote_end_byte']], on='quote_id', how='left')
    
    print(f"Found {len(target_df)} quotes to analyze.")
    
    # Random sample 10 to output to the console for the agent to read
    sample_size = min(10, len(target_df))
    sampled = target_df.sample(sample_size, random_state=42)
    
    print("\n--- SAMPLE FAILURES FOR TAXONOMY ---")
    
    for _, row in sampled.iterrows():
        novel = row['novel']
        novel_txt_path = Path(f"data/raw/pdnc/data/{novel}/{novel}.txt")
        if not novel_txt_path.exists():
            # some pdnc paths have .txt inside novel directory, let's just find the first .txt file
            txt_files = list(Path(f"data/raw/pdnc/data/{novel}").glob("*.txt"))
            if txt_files:
                novel_txt_path = txt_files[0]
                
        text = "TEXT NOT FOUND"
        if novel_txt_path.exists():
            with open(novel_txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Use a window around the quote for context
                start = int(row['quote_start_byte'])
                end = int(row['quote_end_byte'])
                ctx_start = max(0, start - 100)
                ctx_end = min(len(content), end + 100)
                text = f"...{content[ctx_start:start]}>>>{content[start:end]}<<<{content[end:ctx_end]}..."
        
        print(f"\nQuote ID: {row['quote_id']} ({row['novel']})")
        print(f"Category: {row['category']}, Failure Type: {row['failure_type']}")
        print(f"Quote Text with Context: {text.strip()}")
        print(f"Gold Candidate (which failed to rank 1): {row['gold_candidate']}")
        print(f"Baseline Rank: {row['baseline_rank']}, Exp Rank: {row['exp_rank']}")

if __name__ == "__main__":
    analyze()
