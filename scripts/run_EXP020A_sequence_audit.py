import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

def main():
    print("Running EXP020A Sequence Consistency Audit...")
    
    os.makedirs("results/EXP020A", exist_ok=True)
    
    # 1. Load predictions
    preds_df = pd.read_csv("results/EXP016/fully_autoregressive/predictions.csv")
    
    # Need quote types
    q_info_dir = Path("data/raw/pdnc/data")
    type_mappings = {}
    for novel in preds_df['novel'].unique():
        q_info_path = q_info_dir / novel / "quotation_info.csv"
        if q_info_path.exists():
            q_info = pd.read_csv(q_info_path)
            for idx, row in q_info.iterrows():
                q_id_raw = row.get("quoteID")
                if pd.isna(q_id_raw) or not q_id_raw:
                    q_id = f"{novel}_{idx}"
                else:
                    quote_num = str(q_id_raw).strip()
                    if quote_num.startswith('Q'):
                        quote_num = quote_num[1:]
                    q_id = f"{novel}_{quote_num}"
                type_mappings[q_id] = row.get("quoteType")
                
    preds_df['quote_type'] = preds_df['quote_id'].map(type_mappings)
    
    # Group by quote to get ranks and margins
    quote_results = []
    
    def get_q_idx(q_id):
        try: return int(q_id.split('_')[-1])
        except: return 0
        
    for novel, novel_df in preds_df.groupby('novel'):
        quotes = novel_df['quote_id'].unique()
        quotes = sorted(quotes, key=get_q_idx)
        
        gold_speakers = []
        for n, q_id in enumerate(quotes):
            q_df = novel_df[novel_df['quote_id'] == q_id].copy()
            # Sort by score descending
            q_df = q_df.sort_values(by='score', ascending=False).reset_index(drop=True)
            q_df['rank'] = q_df.index + 1
            
            pred_row = q_df.iloc[0]
            pred_speaker = pred_row['candidate']
            pred_prob = pred_row['score']
            
            gold_speaker = q_df['gold_speaker'].iloc[0]
            gold_speakers.append(gold_speaker)
            
            gold_row = q_df[q_df['candidate'] == gold_speaker]
            if len(gold_row) > 0:
                g_rank = gold_row.iloc[0]['rank']
                g_prob = gold_row.iloc[0]['score']
            else:
                g_rank = 999
                g_prob = 0.0
                
            margin = pred_prob - g_prob
            correct = int(pred_speaker == gold_speaker)
            
            # Deterministic transition checks (based on gold history)
            is_persistence = False
            is_alternation = False
            
            if n >= 1:
                if gold_speaker == gold_speakers[n-1]:
                    is_persistence = True
            if n >= 2:
                if gold_speaker == gold_speakers[n-2] and gold_speakers[n-1] != gold_speaker:
                    is_alternation = True
                    
            transition_recoverable = (is_persistence or is_alternation)
            
            quote_results.append({
                "novel": novel,
                "quote_id": q_id,
                "quote_type": q_df['quote_type'].iloc[0],
                "gold_speaker": gold_speaker,
                "prediction": pred_speaker,
                "correct": correct,
                "gold_rank": g_rank,
                "gold_probability": g_prob,
                "prediction_probability": pred_prob,
                "probability_margin": margin,
                "is_persistence_case": is_persistence,
                "is_alternation_case": is_alternation,
                "transition_recoverable": transition_recoverable
            })
            
    res_df = pd.DataFrame(quote_results)
    
    # Analysis 1 & 2: Error Classification
    errors = res_df[res_df['correct'] == 0].copy()
    
    # Output error recoverability
    error_output = errors[['quote_id', 'quote_type', 'gold_rank', 'probability_margin', 
                           'is_persistence_case', 'is_alternation_case', 'transition_recoverable']]
    error_output.to_csv("results/EXP020A/error_recoverability.csv", index=False)
    
    # Decision Gate 1: Ranking Recoverability
    # >= 30% of EXP014 AR errors must have gold speaker in Top-3 with probability margin <= 0.25
    gate1_mask = (errors['gold_rank'] <= 3) & (errors['probability_margin'] <= 0.25)
    gate1_pct = gate1_mask.mean() * 100
    gate1_pass = gate1_pct >= 30.0
    
    print(f"Gate 1: {gate1_pct:.2f}% of errors have gold in Top-3 with margin <= 0.25 (Pass: {gate1_pass})")
    
    # Decision Gate 2: Transition Recoverability
    # For implicit errors: transition_recoverable_implicit_errors / total_implicit_errors >= 20%
    implicit_errors = errors[errors['quote_type'] != 'Explicit']
    gate2_pct = implicit_errors['transition_recoverable'].mean() * 100 if len(implicit_errors) > 0 else 0
    gate2_pass = gate2_pct >= 20.0
    
    print(f"Gate 2: {gate2_pct:.2f}% of implicit errors are transition-recoverable (Pass: {gate2_pass})")
    
    # Analysis 3: Transition Pattern Audit
    # Measure P(current = previous) and ABA frequency over ALL quotes (gold sequence)
    p_persistence = res_df['is_persistence_case'].mean()
    p_alternation = res_df['is_alternation_case'].mean()
    
    pd.DataFrame([{
        "P(current_speaker = previous_speaker)": p_persistence,
        "ABA_alternation_frequency": p_alternation
    }]).to_csv("results/EXP020A/transition_statistics.csv", index=False)
    
    # Analysis 4: Oracle Sequence Ceiling
    # If all transition-recoverable implicit errors were fixed
    current_acc = res_df['correct'].mean()
    
    res_df['oracle_correct'] = res_df['correct']
    mask = (res_df['correct'] == 0) & (res_df['quote_type'] != 'Explicit') & (res_df['transition_recoverable'] == True)
    res_df.loc[mask, 'oracle_correct'] = 1
    
    max_acc = res_df['oracle_correct'].mean()
    
    with open("results/EXP020A/oracle_sequence_ceiling.json", "w") as f:
        json.dump({
            "current_accuracy": current_acc,
            "max_transition_accuracy": max_acc,
            "gate1_passed": bool(gate1_pass),
            "gate2_passed": bool(gate2_pass)
        }, f, indent=4)
        
    print(f"Oracle Ceiling: {current_acc:.4f} -> {max_acc:.4f}")
    
if __name__ == "__main__":
    main()
