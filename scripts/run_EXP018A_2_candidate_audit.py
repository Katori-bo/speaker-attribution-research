import pandas as pd
import numpy as np
from pathlib import Path
import os
from src.utils.logger import setup_logging, get_logger

logger = get_logger("EXP018A.2_CandidateAudit")

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
    logger.info("Starting EXP018A.2 Candidate Recall Audit...")
    
    tf_preds_path = Path("results/EXP016/teacher_forced/predictions.csv")
    if not tf_preds_path.exists():
        logger.error(f"Cannot find {tf_preds_path}")
        return
        
    df = pd.read_csv(tf_preds_path)
    
    # Each row is a candidate for a quote
    grouped = df.groupby('quote_id')
    
    total_evaluated_quotes = len(grouped)
    gold_present = 0
    
    quote_results = []
    
    for quote_id, group in grouped:
        gold_speaker = group['gold_speaker'].iloc[0]
        novel = group['novel'].iloc[0]
        candidates = group['candidate'].tolist()
        
        has_gold = gold_speaker in candidates
        if has_gold:
            gold_present += 1
            sorted_group = group.sort_values(by='score', ascending=False)
            gold_rank = sorted_group['candidate'].tolist().index(gold_speaker) + 1
        else:
            gold_rank = -1
            
        quote_results.append({
            'quote_id': quote_id,
            'novel': novel,
            'has_gold': has_gold,
            'gold_rank': gold_rank
        })
        
    results_df = pd.DataFrame(quote_results)
    
    # 1. Candidate recall
    recall = (gold_present / total_evaluated_quotes) * 100
    print(f"\nTotal evaluated quotes: {total_evaluated_quotes}")
    print(f"Gold present: {gold_present}")
    print(f"Missing: {total_evaluated_quotes - gold_present}")
    print(f"Candidate Recall / Oracle Ceiling: {recall:.2f}%\n")
    
    # 3. Quote-type candidate recall
    novels = results_df['novel'].unique()
    qt_map = build_quote_type_map(novels)
    results_df['quote_type'] = results_df['quote_id'].map(qt_map)
    
    print("Recall by Quote Type:")
    for qt in ['Explicit', 'Anaphoric', 'Implicit']:
        qt_df = results_df[results_df['quote_type'] == qt]
        if not qt_df.empty:
            qt_recall = (qt_df['has_gold'].sum() / len(qt_df)) * 100
            print(f"  {qt}: {qt_recall:.2f}% ({qt_df['has_gold'].sum()} / {len(qt_df)})")
    print()
            
    # 4. Beam pruning conditional on candidate availability
    print("Beam pruning conditional on candidate availability (Teacher Forced Scores):")
    gold_avail_df = results_df[results_df['has_gold']]
    total_avail = len(gold_avail_df)
    
    for k in [1, 3, 5, 10, 20]:
        survives = (gold_avail_df['gold_rank'] <= k).sum()
        pruned = total_avail - survives
        print(f"  Top-{k} Beam:")
        print(f"    Survives: {survives}")
        print(f"    Pruned:   {pruned} ({(pruned/total_avail)*100:.1f}%)")
        
    # Save results
    out_dir = Path("results/EXP018A_2")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "candidate_recall.csv", index=False)
    print(f"\nSaved detailed quote-level recall to {out_dir}/candidate_recall.csv")

if __name__ == "__main__":
    main()
