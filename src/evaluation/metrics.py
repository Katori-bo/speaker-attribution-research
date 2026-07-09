import os
import pandas as pd
import numpy as np

def compute_overall_performance(dfs):
    results = []
    for mode_name, df in dfs.items():
        df['rank'] = df.groupby('quote_id')['score'].rank(ascending=False, method='first')
        acc = df[df['rank'] == 1]['label'].mean()
        results.append({"Mode": mode_name, "Accuracy": f"{acc:.4f}"})
    return pd.DataFrame(results)

def compute_quote_type_accuracy(dfs, quote_type_map, baseline_df=None):
    """Returns accuracy, N counts, and optionally recovery/regression vs baseline per quote type per mode."""
    results = []
    
    if baseline_df is not None:
        baseline_df = baseline_df.copy()
        baseline_df['rank'] = baseline_df.groupby('quote_id')['score'].rank(ascending=False, method='first')
        baseline_top = baseline_df[baseline_df['rank'] == 1].set_index('quote_id')
        baseline_correct = (baseline_top['label'] == 1)
        
    for mode_name, df in dfs.items():
        df['rank'] = df.groupby('quote_id')['score'].rank(ascending=False, method='first')
        top = df[df['rank'] == 1].copy()
        top['quote_type'] = top['quote_id'].map(quote_type_map)
        
        accs = top.groupby('quote_type')['label'].mean()
        counts = top.groupby('quote_type')['label'].count()
        row = {"Mode": mode_name}
        
        # If baseline is provided, compute recovery stats
        if baseline_df is not None:
            top_idx = top.set_index('quote_id')
            variant_correct = (top_idx['label'] == 1)
            
            # Combine to track changes
            compare_df = pd.DataFrame({
                'base_correct': baseline_correct,
                'var_correct': variant_correct,
                'quote_type': top_idx['quote_type']
            })
            
            recovered = compare_df[(~compare_df['base_correct']) & compare_df['var_correct']].groupby('quote_type').size()
            regressed = compare_df[compare_df['base_correct'] & (~compare_df['var_correct'])].groupby('quote_type').size()
            
        for qt in ['Explicit Named', 'Explicit Nominal', 'Explicit Pronoun', 'Explicit', 'Anaphoric', 'Implicit']:
            if qt in accs:
                row[qt] = f"{accs[qt]:.4f}"
                row[f"{qt}_N"] = int(counts[qt])
                if baseline_df is not None:
                    rec = int(recovered.get(qt, 0))
                    reg = int(regressed.get(qt, 0))
                    row[f"{qt}_Recovered"] = rec
                    row[f"{qt}_Regressed"] = reg
                    row[f"{qt}_Net"] = rec - reg
                    
        results.append(row)
    return pd.DataFrame(results)

def compute_conversation_length_accuracy(dfs):
    results = []
    for mode_name, df in dfs.items():
        df['rank'] = df.groupby('quote_id')['score'].rank(ascending=False, method='first')
        top = df[df['rank'] == 1].copy()
        
        # Bin conversation lengths
        def bin_len(x):
            if x >= 10: return "10+"
            return str(int(x))
        top['len_bin'] = top['conversation_length'].apply(bin_len)
        
        accs = top.groupby('len_bin')['label'].mean()
        counts = top.groupby('len_bin')['label'].count()
        
        for k, v in accs.items():
            results.append({
                "Mode": mode_name,
                "Length": k,
                "Accuracy": f"{v:.4f}",
                "Count": counts[k]
            })
    return pd.DataFrame(results)
    
def compute_consecutive_error_runs(df):
    """
    Computes consecutive error runs — sequences of adjacent wrong predictions.
    
    In teacher-forced mode, these represent intrinsically hard sequences (e.g.,
    three ambiguous same-gender pronoun quotes in a row). The model's errors
    cannot corrupt the state by construction.
    
    In autoregressive modes, these include genuine cascades where errors propagate
    through corrupted state. The difference (autoregressive - teacher_forced) isolates
    runs attributable to state corruption.
    """
    df['rank'] = df.groupby('quote_id')['score'].rank(ascending=False, method='first')
    top = df[df['rank'] == 1].copy()
    
    run_lengths = []
    recovery_lengths = []
    
    for novel, novel_df in top.groupby('novel'):
        def get_qidx(x): return int(x.split('_')[-1])
        novel_df = novel_df.sort_values(by='quote_id', key=lambda col: col.map(get_qidx)).reset_index(drop=True)
        
        current_run = 0
        in_error_state = False
        quotes_since_error = 0
        
        for _, row in novel_df.iterrows():
            correct = (row['label'] == 1)
            
            if not correct:
                current_run += 1
                in_error_state = True
                quotes_since_error += 1
            else:
                if current_run > 0:
                    run_lengths.append(current_run)
                    current_run = 0
                if in_error_state:
                    recovery_lengths.append(quotes_since_error)
                    in_error_state = False
                    quotes_since_error = 0
                    
        # If novel ends in error state
        if current_run > 0:
            run_lengths.append(current_run)
            
    avg_run = np.mean(run_lengths) if run_lengths else 0.0
    avg_recovery = np.mean(recovery_lengths) if recovery_lengths else 0.0
    
    return {
        "Total Error Runs": len(run_lengths),
        "Average Run Length": f"{avg_run:.2f}",
        "Max Run Length": max(run_lengths) if run_lengths else 0,
        "Average Recovery Length": f"{avg_recovery:.2f}",
        "Max Recovery Length": max(recovery_lengths) if recovery_lengths else 0
    }

# Keep old name as alias for backwards compatibility
compute_cascade_and_recovery = compute_consecutive_error_runs

def compute_confidence_baseline(df, mode_name=""):
    """
    Computes baseline confidence statistics: mean confidence on correct vs wrong 
    predictions. Essential for contextualizing drift analysis.
    """
    df['rank'] = df.groupby('quote_id')['score'].rank(ascending=False, method='first')
    top = df[df['rank'] == 1].copy()
    
    correct_mask = top['label'] == 1
    
    return {
        "Mode": mode_name,
        "Mean_Confidence_Correct": f"{top.loc[correct_mask, 'score'].mean():.4f}",
        "Mean_Confidence_Wrong": f"{top.loc[~correct_mask, 'score'].mean():.4f}",
        "Calibration_Gap_pp": f"{(top.loc[correct_mask, 'score'].mean() - top.loc[~correct_mask, 'score'].mean()) * 100:.1f}",
        "N_Correct": int(correct_mask.sum()),
        "N_Wrong": int((~correct_mask).sum())
    }

def compute_confidence_drift(df):
    """
    Tracks prediction probability up to k steps after an error to see if model is blindly confident.
    
    NOTE: This table has survivorship bias — it only includes quotes that recovered
    within the tracking window. Long cascades (up to 19 quotes) that never recover
    are excluded, biasing confidence numbers toward cleanly-resolved cases.
    """
    df['rank'] = df.groupby('quote_id')['score'].rank(ascending=False, method='first')
    top = df[df['rank'] == 1].copy()
    
    drift_data = []
    
    for novel, novel_df in top.groupby('novel'):
        def get_qidx(x): return int(x.split('_')[-1])
        novel_df = novel_df.sort_values(by='quote_id', key=lambda col: col.map(get_qidx)).reset_index(drop=True)
        
        error_idx = -1
        for i, row in novel_df.iterrows():
            correct = (row['label'] == 1)
            if not correct:
                error_idx = i
                drift_data.append({"distance": 0, "confidence": row['score'], "correct": 0})
            elif error_idx != -1:
                dist = i - error_idx
                if dist <= 5: # track up to 5 steps
                    drift_data.append({"distance": dist, "confidence": row['score'], "correct": 1})
                else:
                    error_idx = -1 # reset tracking after 5 correct steps
                    
    drift_df = pd.DataFrame(drift_data)
    if drift_df.empty: return pd.DataFrame()
    
    res = drift_df.groupby('distance').agg(
        Average_Confidence=('confidence', 'mean'),
        Accuracy=('correct', 'mean'),
        Count=('correct', 'count')
    ).reset_index()
    
    return res

def compute_anchor_state_drift_stats(df):
    """
    Computes EXP017C explicit anchor state drift statistics.
    Checks if state drifted before anchor, and if anchor overrides correctly.
    """
    df['rank'] = df.groupby('quote_id')['score'].rank(ascending=False, method='first')
    top = df[df['rank'] == 1].copy()
    
    anchor_events = 0
    state_drifted_at_anchor = 0
    state_correct_at_anchor = 0
    state_resets_applied = 0
    drifted_that_would_reset = 0
    
    for i, row in top.iterrows():
        if row.get('is_anchor_fired', 0) == 1:
            anchor_events += 1
            if row.get('state_drifted', False):
                state_drifted_at_anchor += 1
                if row.get('persisted_last_speaker') != row.get('anchor_attributed_speaker'):
                    drifted_that_would_reset += 1
            else:
                state_correct_at_anchor += 1
                
            if row.get('state_reset_applied', False):
                state_resets_applied += 1
                
    return {
        "anchor_events": anchor_events,
        "state_drifted_at_anchor": state_drifted_at_anchor,
        "state_correct_at_anchor": state_correct_at_anchor,
        "state_resets_applied": state_resets_applied,
        "drifted_that_would_reset": drifted_that_would_reset
    }

def compute_post_anchor_window_accuracy(ear_df, fa_df, qt_map, k=[1, 3, 5]):
    """
    Computes accuracy on implicit quotes in the window following an anchor.
    Compares EAR vs FA at matched positions.
    """
    ear_df['rank'] = ear_df.groupby('quote_id')['score'].rank(ascending=False, method='first')
    ear_top = ear_df[ear_df['rank'] == 1].copy()
    
    fa_df['rank'] = fa_df.groupby('quote_id')['score'].rank(ascending=False, method='first')
    fa_top = fa_df[fa_df['rank'] == 1].copy().set_index('quote_id')
    
    ear_top['quote_type'] = ear_top['quote_id'].map(qt_map)
    fa_top['quote_type'] = fa_top.index.map(qt_map)
    
    results = []
    
    for novel, novel_df in ear_top.groupby('novel'):
        def get_qidx(x): return int(x.split('_')[-1])
        novel_df = novel_df.sort_values(by='quote_id', key=lambda col: col.map(get_qidx)).reset_index(drop=True)
        
        for i, row in novel_df.iterrows():
            if row.get('is_anchor_fired', 0) == 1:
                state_drifted = row.get('state_drifted', False)
                
                for window in k:
                    for j in range(1, window + 1):
                        if i + j < len(novel_df):
                            post_row = novel_df.iloc[i+j]
                            if post_row['quote_type'] == 'Implicit':
                                q_id = post_row['quote_id']
                                ear_correct = int(post_row['label'] == 1)
                                fa_correct = int(fa_top.loc[q_id]['label'] == 1) if q_id in fa_top.index else None
                                
                                results.append({
                                    "novel": novel,
                                    "anchor_position": i,
                                    "target_position": i+j,
                                    "window": window,
                                    "state_drifted": state_drifted,
                                    "quote_id": q_id,
                                    "ear_correct": ear_correct,
                                    "fa_correct": fa_correct
                                })
                                
    return pd.DataFrame(results)

def compute_confidence_at_error_inclusive(df):
    """
    Computes confidence at error for all errors, regardless of recovery.
    """
    df['rank'] = df.groupby('quote_id')['score'].rank(ascending=False, method='first')
    top = df[df['rank'] == 1].copy()
    
    errors = top[top['label'] != 1]
    
    return pd.DataFrame([{
        "Total_Errors": len(errors),
        "Mean_Confidence_At_Error_Inclusive": f"{errors['score'].mean():.4f}" if len(errors) > 0 else "0.0000"
    }])

def compute_cascade_survivorship(df):
    """
    Computes cascade survivorship stats (how many error cascades never recover within 5 steps).
    """
    df['rank'] = df.groupby('quote_id')['score'].rank(ascending=False, method='first')
    top = df[df['rank'] == 1].copy()
    
    total_errors = 0
    recovered_within_5 = 0
    unrecovered = 0
    
    for novel, novel_df in top.groupby('novel'):
        def get_qidx(x): return int(x.split('_')[-1])
        novel_df = novel_df.sort_values(by='quote_id', key=lambda col: col.map(get_qidx)).reset_index(drop=True)
        
        error_idx = -1
        for i, row in novel_df.iterrows():
            correct = (row['label'] == 1)
            if not correct:
                if error_idx == -1:
                    error_idx = i
                    total_errors += 1
            else:
                if error_idx != -1:
                    dist = i - error_idx
                    if dist <= 5:
                        recovered_within_5 += 1
                    else:
                        unrecovered += 1
                    error_idx = -1
                    
        if error_idx != -1:
            unrecovered += 1
            
    pct_excluded = (unrecovered / total_errors) * 100 if total_errors > 0 else 0
    
    return {
        "total_errors": total_errors,
        "recovered_within_5": recovered_within_5,
        "unrecovered": unrecovered,
        "pct_excluded_from_drift_table": f"{pct_excluded:.1f}%"
    }
