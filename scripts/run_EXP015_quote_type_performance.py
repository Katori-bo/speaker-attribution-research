import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from src.utils.config import get_data_dir

def get_ranking_metrics(y_true, y_score, groups):
    df = pd.DataFrame({'label': y_true, 'score': y_score, 'group': groups})
    df['rank'] = df.groupby('group')['score'].rank(ascending=False, method='first')
    top_preds = df[df['rank'] == 1]
    return top_preds['label'].mean()

def main():
    exp014_cache = get_data_dir() / "phase2" / "candidate_features_exp014.csv"
    if not exp014_cache.exists():
        print("Error: EXP014 features cache not found!")
        return
        
    df = pd.read_csv(exp014_cache)
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    # Feature Sets
    top3_feats = ['candidate_is_explicit_mention', 'candidate_is_previous_speaker', 'candidate_is_recent_mention']
    
    base_feats = [c for c in df.columns if c not in [
        "quote_id", "novel", "candidate", "gold_speaker", "split", "label",
        "quote_start_byte", "quote_end_byte", "quoteByteSpans",
        "candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency",
        "candidate_is_attributed_speaker"
    ] and not c.startswith("symbolic_")]
    
    exp012_feats = base_feats + ["candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency"]
    exp014_feats = exp012_feats + ["candidate_is_attributed_speaker"]
    
    # Re-predict using same models to get consistent scores
    lr = LogisticRegression(random_state=42, class_weight='balanced')
    lr.fit(train_df[top3_feats], train_df['label'])
    test_df['score_top3'] = lr.predict_proba(test_df[top3_feats])[:, 1]
    
    gbm_012 = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    gbm_012.fit(train_df[exp012_feats], train_df['label'])
    test_df['score_exp012'] = gbm_012.predict_proba(test_df[exp012_feats])[:, 1]
    
    gbm_014 = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    gbm_014.fit(train_df[exp014_feats], train_df['label'])
    test_df['score_exp014'] = gbm_014.predict_proba(test_df[exp014_feats])[:, 1]
    
    # Categorize Quote Types using PDNC Gold dataset annotations
    quote_types_dict = {}
    booknlp_correct_map = {} # quote_id -> bool
    
    for novel in test_df['novel'].unique():
        q_info_path = get_data_dir() / "data" / novel / "quotation_info.csv"
        
        ndf = test_df[test_df['novel'] == novel]
        quote_df = ndf.groupby('quote_id').first().reset_index()
        
        if not q_info_path.exists():
            print(f"Missing quotation info for {novel}")
            continue
            
        pdnc_quotes = pd.read_csv(q_info_path)
        
        # Build mapping from Q# to Type
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
                final_type = 'Implicit' # Fallback
                
            novel_type_map[row['quoteID']] = final_type
            
        # BookNLP accuracy check
        quotes_file = get_data_dir() / "booknlp_out" / novel / f"{novel}.quotes"
        tokens_file = get_data_dir() / "booknlp_out" / novel / f"{novel}.tokens"
        
        has_booknlp = quotes_file.exists() and tokens_file.exists()
        if has_booknlp:
            import csv
            b_quotes = pd.read_csv(quotes_file, sep='\t', quoting=csv.QUOTE_NONE)
            b_tokens = pd.read_csv(tokens_file, sep='\t', quoting=csv.QUOTE_NONE)
            token_bytes = b_tokens.set_index('token_ID_within_document')[['byte_onset', 'byte_offset']]
            
        novel_matches = []
        for _, gq in quote_df.iterrows():
            g_start = int(gq['quote_start_byte'])
            g_end = int(gq['quote_end_byte'])
            q_id = gq['quote_id']
            
            # Map type
            pdnc_id = f"Q{q_id.split('_')[-1]}"
            quote_types_dict[q_id] = novel_type_map.get(pdnc_id, 'Implicit')
            
            if has_booknlp:
                best_match = None
                best_overlap = 0
                for _, bq in b_quotes.iterrows():
                    start_tok = bq['quote_start']
                    end_tok = bq['quote_end']
                    if start_tok in token_bytes.index and end_tok in token_bytes.index:
                        b_start = token_bytes.loc[start_tok, 'byte_onset']
                        b_end = token_bytes.loc[end_tok, 'byte_offset']
                        overlap = min(b_end, g_end) - max(b_start, g_start)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match = bq
                if best_match is not None:
                    novel_matches.append({
                        'quote_id': q_id,
                        'char_id': best_match['char_id'],
                        'gold_speaker': gq['gold_speaker']
                    })
                else:
                    booknlp_correct_map[q_id] = False
            else:
                booknlp_correct_map[q_id] = False
                
        if novel_matches:
            match_df = pd.DataFrame(novel_matches)
            char_map = match_df.groupby('char_id')['gold_speaker'].agg(lambda x: x.value_counts().index[0]).to_dict()
            for _, row in match_df.iterrows():
                is_correct = (char_map.get(row['char_id']) == row['gold_speaker'])
                booknlp_correct_map[row['quote_id']] = is_correct

    test_df['quote_type'] = test_df['quote_id'].map(quote_types_dict)
    
    out_rows = []
    systems = ['BookNLP', 'Top3', 'EXP012', 'EXP014']
    res_table = {sys: {} for sys in systems}
    
    type_categories = ['Explicit Named', 'Explicit Nominal', 'Explicit Pronoun', 'Implicit']
    
    for qtype in type_categories:
        q_ids = test_df[test_df['quote_type'] == qtype]['quote_id'].unique()
        type_df = test_df[test_df['quote_id'].isin(q_ids)]
        count = len(q_ids)
        
        print(f"{qtype}: {count} quotes")
        if count == 0:
            for sys in systems:
                res_table[sys][qtype] = 0.0
            continue
            
        acc_top3 = get_ranking_metrics(type_df['label'], type_df['score_top3'], type_df['quote_id'])
        res_table['Top3'][qtype] = acc_top3
        
        acc_012 = get_ranking_metrics(type_df['label'], type_df['score_exp012'], type_df['quote_id'])
        res_table['EXP012'][qtype] = acc_012
        
        acc_014 = get_ranking_metrics(type_df['label'], type_df['score_exp014'], type_df['quote_id'])
        res_table['EXP014'][qtype] = acc_014
        
        bnlp_correct = sum(booknlp_correct_map.get(qid, False) for qid in q_ids)
        acc_bnlp = bnlp_correct / count if count > 0 else 0
        res_table['BookNLP'][qtype] = acc_bnlp
        
    final_rows = []
    for sys in systems:
        final_rows.append({
            'System': sys,
            'Explicit Named': f"{res_table[sys].get('Explicit Named', 0):.4f}",
            'Explicit Nominal': f"{res_table[sys].get('Explicit Nominal', 0):.4f}",
            'Explicit Pronoun': f"{res_table[sys].get('Explicit Pronoun', 0):.4f}",
            'Implicit': f"{res_table[sys].get('Implicit', 0):.4f}"
        })
        
    os.makedirs("results/EXP015", exist_ok=True)
    res_df = pd.DataFrame(final_rows)
    res_df.to_csv("results/EXP015/quote_type_performance.csv", index=False)
    
    print("\nResults:")
    print(res_df.to_string(index=False))
    
    implicit_acc = res_table['EXP014'].get('Implicit', 0)
    print(f"\nMissing implicit accuracy reported for EXP014: {implicit_acc:.2%}")

if __name__ == "__main__":
    main()
