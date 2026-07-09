import pandas as pd
import json

def get_gate1_pct(preds_path):
    preds_df = pd.read_csv(preds_path)
    
    # Sort quotes
    def get_q_idx(q_id):
        try: return int(q_id.split('_')[-1])
        except: return 0
        
    quote_results = []
    for novel, novel_df in preds_df.groupby('novel'):
        quotes = sorted(novel_df['quote_id'].unique(), key=get_q_idx)
        for q_id in quotes:
            q_df = novel_df[novel_df['quote_id'] == q_id].copy()
            q_df = q_df.sort_values(by='score', ascending=False).reset_index(drop=True)
            q_df['rank'] = q_df.index + 1
            
            pred_row = q_df.iloc[0]
            pred_speaker = pred_row['candidate']
            pred_prob = pred_row['score']
            
            gold_speaker = q_df['gold_speaker'].iloc[0]
            
            gold_row = q_df[q_df['candidate'] == gold_speaker]
            if len(gold_row) > 0:
                g_rank = gold_row.iloc[0]['rank']
                g_prob = gold_row.iloc[0]['score']
            else:
                g_rank = 999
                g_prob = 0.0
                
            margin = pred_prob - g_prob
            correct = int(pred_speaker == gold_speaker)
            
            quote_results.append({
                "quote_id": q_id,
                "correct": correct,
                "gold_rank": g_rank,
                "probability_margin": margin
            })
            
    res_df = pd.DataFrame(quote_results)
    errors = res_df[res_df['correct'] == 0]
    gate1_mask = (errors['gold_rank'] <= 3) & (errors['probability_margin'] <= 0.25)
    gate1_pct = gate1_mask.mean() * 100
    
    # Also evaluate AR errors under this state
    return gate1_pct, res_df

def main():
    print("Running EXP020A.1 State Corruption Diagnostic...")
    
    ar_pct, ar_res = get_gate1_pct("results/EXP016/fully_autoregressive/predictions.csv")
    print(f"AR State - Gate 1 (Top3 close errors): {ar_pct:.2f}%")
    
    tf_pct, tf_res = get_gate1_pct("results/EXP016/teacher_forced/predictions.csv")
    print(f"TF State - Gate 1 (Top3 close errors): {tf_pct:.2f}%")
    
    # What happens to the AR errors under TF state?
    ar_errors = set(ar_res[ar_res['correct'] == 0]['quote_id'])
    tf_on_ar_errors = tf_res[tf_res['quote_id'].isin(ar_errors)]
    
    # How many of these are Top 3 close OR correct under TF?
    # Correct means margin <= 0 (it is 0), and rank = 1, so it satisfies rank<=3 and margin<=0.25!
    tf_on_ar_mask = (tf_on_ar_errors['gold_rank'] <= 3) & (tf_on_ar_errors['probability_margin'] <= 0.25)
    availability_pct = tf_on_ar_mask.mean() * 100
    print(f"AR Errors under TF State - Top3 Availability: {availability_pct:.2f}%")

if __name__ == "__main__":
    main()
