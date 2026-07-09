import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import setup_logging, get_logger

logger = get_logger("EXP018A_Statistics")

def compute_beam_recovery_regression(k1_df, kx_df):
    """Compares baseline (K=1) to K=X to find recovered and regressed quotes."""
    # Ensure both dataframes are aligned
    k1_df = k1_df.set_index('quote_id')
    kx_df = kx_df.set_index('quote_id')
    
    # We only care about intersecting quotes
    intersect = k1_df.index.intersection(kx_df.index)
    k1 = k1_df.loc[intersect]
    kx = kx_df.loc[intersect]
    
    k1_correct = (k1['candidate'] == k1['gold_speaker'])
    kx_correct = (kx['candidate'] == kx['gold_speaker'])
    
    recovered_mask = (~k1_correct) & kx_correct
    regressed_mask = k1_correct & (~kx_correct)
    
    total_recovered = recovered_mask.sum()
    total_regressed = regressed_mask.sum()
    
    overall = {
        'recovered': total_recovered,
        'regressed': total_regressed,
        'net': total_recovered - total_regressed
    }
    
    # Split by quote_type if available
    by_type = []
    if 'quote_type' in k1.columns:
        for qt in k1['quote_type'].unique():
            qt_mask = (k1['quote_type'] == qt)
            rec = (recovered_mask & qt_mask).sum()
            reg = (regressed_mask & qt_mask).sum()
            by_type.append({
                'quote_type': qt,
                'recovered': rec,
                'regressed': reg,
                'net': rec - reg
            })
            
    return overall, pd.DataFrame(by_type)

def run_paired_bootstrap(k1_correct, kx_correct, n_iterations=10000):
    np.random.seed(42)
    deltas = []
    n_samples = len(k1_correct)
    
    # Convert to numpy arrays for faster sampling
    k1_arr = k1_correct.values
    kx_arr = kx_correct.values
    
    for _ in range(n_iterations):
        indices = np.random.randint(0, n_samples, n_samples)
        k1_sample = k1_arr[indices]
        kx_sample = kx_arr[indices]
        
        k1_acc = k1_sample.mean()
        kx_acc = kx_sample.mean()
        
        deltas.append(kx_acc - k1_acc)
        
    deltas = np.array(deltas)
    mean_delta = deltas.mean()
    ci_low = np.percentile(deltas, 2.5)
    ci_high = np.percentile(deltas, 97.5)
    
    # Check if 0 is outside the interval
    significant = not (ci_low <= 0 <= ci_high)
    
    return mean_delta * 100, ci_low * 100, ci_high * 100, significant

def compute_oracle_decay(oracle_logs_df):
    """Computes oracle decay by absolute quote position and relative progress."""
    # oracle_logs_df has: novel, quote_id, idx_in_novel, total_quotes_in_novel, gold_survived, death_reason
    decay_by_position = {}
    decay_by_progress = {}
    
    # Checkpoints for progress: 0%, 10%, 25%, 50%, 75%, 100%
    checkpoints = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
    
    novels = oracle_logs_df['novel'].unique()
    n_novels = len(novels)
    
    # Calculate for absolute positions: 1, 10, 50, 100, 500, etc.
    positions = [1, 10, 50, 100, 200, 500, 1000]
    
    for pos in positions:
        survived = 0
        for novel in novels:
            novel_logs = oracle_logs_df[oracle_logs_df['novel'] == novel]
            if pos <= len(novel_logs):
                # Did it survive at least until pos?
                # The log tells us if it survived at that exact step. 
                # If it's dead, it's dead forever. So we just check the log at pos-1 (since idx is 0-based).
                log_at_pos = novel_logs.iloc[pos-1]
                if log_at_pos['gold_survived']:
                    survived += 1
        decay_by_position[pos] = (survived / n_novels) * 100
        
    for cp in checkpoints:
        survived = 0
        for novel in novels:
            novel_logs = oracle_logs_df[oracle_logs_df['novel'] == novel]
            total_quotes = len(novel_logs)
            target_idx = int(cp * (total_quotes - 1))
            if target_idx < total_quotes:
                log_at_pos = novel_logs.iloc[target_idx]
                if log_at_pos['gold_survived']:
                    survived += 1
        decay_by_progress[cp] = (survived / n_novels) * 100
        
    return decay_by_position, decay_by_progress

def build_quote_type_map(novels):
    qt_map = {}
    q_info_dir = "data/raw/pdnc/data"
    for novel in novels:
        q_info_path = os.path.join(q_info_dir, novel, "quotation_info.csv")
        if not os.path.exists(q_info_path):
            continue
        q_info = pd.read_csv(q_info_path)
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

def main():
    setup_logging()
    logger.info("Starting EXP018A Validation Analysis...")
    
    k_values = [3, 5, 10, 20]
    out_dir = Path("results/EXP018A")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    k1_dir = Path("results/EXP018A/beam_K1")
    if not (k1_dir / "predictions.csv").exists():
        logger.error("K=1 baseline predictions not found. Run the K sweep first.")
        return
        
    k1_df = pd.read_csv(k1_dir / "predictions.csv")
    novels = k1_df['novel'].unique()
    qt_map = build_quote_type_map(novels)
    k1_df['quote_type'] = k1_df['quote_id'].map(qt_map)
    k1_correct = (k1_df['candidate'] == k1_df['gold_speaker'])
    k1_df['is_correct'] = k1_correct
    
    bootstrap_results = []
    recovery_regression = []
    recovery_by_type = []
    
    oracle_decay_pos = {}
    oracle_decay_prog = {}
    oracle_deaths = []
    
    # K1 Oracle Stats
    k1_logs_file = k1_dir / "oracle_logs.csv"
    if k1_logs_file.exists():
        k1_logs = pd.read_csv(k1_logs_file)
        dec_pos, dec_prog = compute_oracle_decay(k1_logs)
        oracle_decay_pos['K1'] = dec_pos
        oracle_decay_prog['K1'] = dec_prog
    
    for k in k_values:
        k_dir = out_dir / f"beam_K{k}"
        pred_file = k_dir / "predictions.csv"
        logs_file = k_dir / "oracle_logs.csv"
        
        if not pred_file.exists():
            continue
            
        kx_df = pd.read_csv(pred_file)
        kx_df['quote_type'] = kx_df['quote_id'].map(qt_map)
        
        # --- Bootstrap Analysis ---
        # Align rows by quote_id
        merged = pd.merge(k1_df[['quote_id', 'is_correct']], kx_df[['quote_id', 'candidate', 'gold_speaker']], on='quote_id')
        k1_c = merged['is_correct']
        kx_c = (merged['candidate'] == merged['gold_speaker'])
        
        mean_delta, ci_low, ci_high, significant = run_paired_bootstrap(k1_c, kx_c)
        bootstrap_results.append({
            'comparison': f'K{k}-K1',
            'mean_delta': f"{mean_delta:+.2f}",
            'ci_low': f"{ci_low:+.2f}",
            'ci_high': f"{ci_high:+.2f}",
            'significant': str(significant).lower()
        })
        
        # --- Recovery/Regression Analysis ---
        overall, by_type = compute_beam_recovery_regression(k1_df, kx_df)
        overall['comparison'] = f'K{k}_vs_K1'
        recovery_regression.append(overall)
        
        if not by_type.empty:
            by_type['comparison'] = f'K{k}_vs_K1'
            recovery_by_type.append(by_type)
            
        # --- Oracle Decay Analysis ---
        if logs_file.exists():
            kx_logs = pd.read_csv(logs_file)
            dec_pos, dec_prog = compute_oracle_decay(kx_logs)
            oracle_decay_pos[f'K{k}'] = dec_pos
            oracle_decay_prog[f'K{k}'] = dec_prog
            
            # Oracle death reason (only for K20, or aggregate?)
            # Let's do it for all K
            deaths = kx_logs[kx_logs['death_reason'].notna()]
            death_counts = deaths['death_reason'].value_counts()
            
            # Only count actual deaths for percentage (not_evaluated is just an event)
            actual_deaths = deaths[deaths['death_reason'].isin(['candidate_missing', 'beam_pruned'])]
            total_actual_deaths = len(actual_deaths)
            
            for reason, count in death_counts.items():
                if reason == 'not_evaluated':
                    pct_str = "N/A"
                else:
                    pct_str = f"{(count/total_actual_deaths)*100:.1f}%" if total_actual_deaths > 0 else "0.0%"
                    
                oracle_deaths.append({
                    'K': k,
                    'reason': reason,
                    'count': count,
                    'percentage': pct_str
                })
                
    # --- Save Artifacts ---
    if bootstrap_results:
        pd.DataFrame(bootstrap_results).to_csv(out_dir / "bootstrap_results.csv", index=False)
        logger.info(f"Saved bootstrap_results.csv")
        
    if recovery_regression:
        pd.DataFrame(recovery_regression).to_csv(out_dir / "recovery_regression.csv", index=False)
        if recovery_by_type:
            pd.concat(recovery_by_type).to_csv(out_dir / "recovery_regression_by_type.csv", index=False)
        logger.info(f"Saved recovery_regression.csv")
        
    if oracle_decay_pos:
        # Format: quote_position, K1, K3, K5, K10, K20
        pos_df = pd.DataFrame(oracle_decay_pos)
        pos_df.index.name = 'quote_position'
        pos_df.reset_index().to_csv(out_dir / "oracle_decay_by_position.csv", index=False)
        
        prog_df = pd.DataFrame(oracle_decay_prog)
        prog_df.index.name = 'progress'
        prog_df.reset_index().to_csv(out_dir / "oracle_decay_by_progress.csv", index=False)
        logger.info(f"Saved oracle decay curves")
        
    if oracle_deaths:
        pd.DataFrame(oracle_deaths).to_csv(out_dir / "oracle_death_analysis.csv", index=False)
        logger.info(f"Saved oracle_death_analysis.csv")

    logger.info("All statistical analyses completed.")

if __name__ == "__main__":
    main()
