import os
import pandas as pd
import numpy as np
from src.utils.config import get_data_dir
from sklearn.metrics import log_loss

def get_ranking_metrics(y_true, y_score, groups):
    df = pd.DataFrame({'label': y_true, 'score': y_score, 'group': groups})
    df['rank'] = df.groupby('group')['score'].rank(ascending=False, method='first')
    top_preds = df[df['rank'] == 1]
    return top_preds['label'].mean()

def compute_ci(accuracy, n):
    import math
    z = 1.96
    margin = z * math.sqrt((accuracy * (1 - accuracy)) / n) if n > 0 else 0
    return max(0, accuracy - margin), min(1, accuracy + margin)

def main():
    os.makedirs("results/EXP015B", exist_ok=True)
    exp014_cache = get_data_dir() / "phase2" / "candidate_features_exp014.csv"
    df = pd.read_csv(exp014_cache)
    
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    # Audit 7: Novel Split Verification
    train_novels = set(train_df['novel'].unique())
    test_novels = set(test_df['novel'].unique())
    intersection = train_novels.intersection(test_novels)
    
    with open("results/EXP015B/novel_split_verification.md", "w") as f:
        f.write("# Audit 7: Novel Split Verification\n\n")
        f.write(f"Train novels ({len(train_novels)}): {', '.join(train_novels)}\n")
        f.write(f"Test novels ({len(test_novels)}): {', '.join(test_novels)}\n\n")
        f.write(f"Intersection: {intersection if intersection else '∅ (No overlap)'}\n\n")
        
        # Verify duplicated quotes
        train_quotes = set(train_df['quote_id'].unique())
        test_quotes = set(test_df['quote_id'].unique())
        quote_intersection = train_quotes.intersection(test_quotes)
        f.write(f"Duplicated Quotes between Train and Test: {len(quote_intersection)}\n")

    # Audit 2: Quote Type Mapping (Reusing logic from earlier)
    quote_types_dict = {}
    
    # Process ALL novels to get global stats
    all_novels = df['novel'].unique()
    for novel in all_novels:
        q_info_path = get_data_dir() / "data" / novel / "quotation_info.csv"
        if not q_info_path.exists(): continue
        
        pdnc_quotes = pd.read_csv(q_info_path)
        novel_type_map = {}
        for _, row in pdnc_quotes.iterrows():
            q_type = str(row['quoteType'])
            m_phrase = str(row['referringExpression'])
            
            if q_type == 'Implicit':
                final_type = 'Implicit'
            elif q_type == 'Anaphoric':
                final_type = 'Explicit Pronoun'
            elif q_type == 'Explicit':
                if any(c.isupper() for c in m_phrase):
                    final_type = 'Explicit Named'
                else:
                    final_type = 'Explicit Nominal'
            else:
                final_type = 'Implicit'
                
            novel_type_map[row['quoteID']] = final_type
            
        ndf = df[df['novel'] == novel]
        for q_id in ndf['quote_id'].unique():
            pdnc_id = f"Q{q_id.split('_')[-1]}"
            quote_types_dict[q_id] = novel_type_map.get(pdnc_id, 'Implicit')
            
    df['quote_type'] = df['quote_id'].map(quote_types_dict)
    test_df['quote_type'] = test_df['quote_id'].map(quote_types_dict)
    
    # Audit 3: Quote Type Distribution (on test set to match EXP015 evaluation scale)
    q_counts = test_df.groupby('quote_id')['quote_type'].first().value_counts()
    total_test = len(test_df['quote_id'].unique())
    
    dist_rows = []
    for qt, count in q_counts.items():
        dist_rows.append({
            'Quote Type': qt,
            'Count': count,
            'Percentage': f"{(count / total_test) * 100:.1f}%"
        })
    pd.DataFrame(dist_rows).to_csv("results/EXP015B/quote_type_distribution.csv", index=False)
    
    # Audit 4: Candidate Set Difficulty
    cand_rows = []
    for qtype in ['Explicit Named', 'Explicit Nominal', 'Explicit Pronoun', 'Implicit']:
        q_df = test_df[test_df['quote_type'] == qtype]
        if q_df.empty: continue
        
        counts = q_df.groupby('quote_id').size()
        cand_rows.append({
            'Quote Type': qtype,
            'Avg Candidates': f"{counts.mean():.2f}",
            'Median': f"{counts.median():.1f}",
            'Max': counts.max()
        })
    pd.DataFrame(cand_rows).to_csv("results/EXP015B/candidate_statistics.csv", index=False)
    
    # Audit 5: Category Stability
    # We need to compute EXP014 predictions again to get LogLoss and Accuracy
    from sklearn.ensemble import HistGradientBoostingClassifier
    base_feats = [c for c in df.columns if c not in [
        "quote_id", "novel", "candidate", "gold_speaker", "split", "label",
        "quote_start_byte", "quote_end_byte", "quoteByteSpans",
        "candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency",
        "candidate_is_attributed_speaker", "quote_type"
    ] and not c.startswith("symbolic_")]
    exp014_feats = base_feats + ["candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency", "candidate_is_attributed_speaker"]
    
    gbm_014 = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    gbm_014.fit(train_df[exp014_feats], train_df['label'])
    test_df['score_exp014'] = gbm_014.predict_proba(test_df[exp014_feats])[:, 1]
    
    stability_rows = []
    for qtype in ['Explicit Named', 'Explicit Nominal', 'Explicit Pronoun', 'Implicit']:
        type_df = test_df[test_df['quote_type'] == qtype]
        q_ids = type_df['quote_id'].unique()
        count = len(q_ids)
        if count == 0: continue
        
        acc = get_ranking_metrics(type_df['label'], type_df['score_exp014'], type_df['quote_id'])
        ci_lower, ci_upper = compute_ci(acc, count)
        
        # LogLoss requires true labels and predicted probabilities for all candidates
        # We compute log loss over the binary predictions
        ll = log_loss(type_df['label'], type_df['score_exp014'])
        
        stability_rows.append({
            'Quote Type': qtype,
            'N': count,
            'Accuracy': f"{acc:.4f}",
            'LogLoss': f"{ll:.4f}",
            '95% CI': f"[{ci_lower:.4f}, {ci_upper:.4f}]"
        })
    pd.DataFrame(stability_rows).to_csv("results/EXP015B/quote_type_counts.csv", index=False)
    print("Audits 3, 4, 5, 7 completed and saved to results/EXP015B/")

if __name__ == "__main__":
    main()
