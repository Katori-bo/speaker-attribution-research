import pandas as pd
import numpy as np
import random
import os
from pathlib import Path
from src.utils.config import get_data_dir
from src.analysis.quote_classifier import QuoteClassifier

def main():
    exp012_cache = get_data_dir() / "phase2" / "candidate_features_exp012.csv"
    df = pd.read_csv(exp012_cache)
    test_df = df[df['split'] == 'test']
    unique_quotes = test_df['quote_id'].unique()
    
    # Get 100 random quotes
    np.random.seed(42)
    sample_quotes = np.random.choice(unique_quotes, size=100, replace=False)
    
    classifier = QuoteClassifier()
    
    results = []
    
    for novel in test_df['novel'].unique():
        quotes_file = get_data_dir() / "booknlp_out" / novel / f"{novel}.quotes"
        tokens_file = get_data_dir() / "booknlp_out" / novel / f"{novel}.tokens"
        if not quotes_file.exists() or not tokens_file.exists():
            continue
            
        import csv
        b_quotes = pd.read_csv(quotes_file, sep='\t', quoting=csv.QUOTE_NONE)
        b_tokens = pd.read_csv(tokens_file, sep='\t', quoting=csv.QUOTE_NONE)
        token_bytes = b_tokens.set_index('token_ID_within_document')[['byte_onset', 'byte_offset', 'word']]
        
        ndf = test_df[(test_df['novel'] == novel) & (test_df['quote_id'].isin(sample_quotes))]
        if ndf.empty:
            continue
            
        gold_df = ndf[ndf['label'] == 1].groupby('quote_id').first().reset_index()
        
        for _, gq in gold_df.iterrows():
            g_start = int(gq['quote_start_byte'])
            g_end = int(gq['quote_end_byte'])
            q_id = gq['quote_id']
            
            # Find best overlapping booknlp quote
            best_match = None
            best_overlap = 0
            
            b_starts = []
            b_ends = []
            
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
                mention_phrase = str(best_match['mention_phrase']) if pd.notnull(best_match['mention_phrase']) else ""
                q_type = classifier.classify(mention_phrase).value
                
                # Get context: some tokens before and after
                start_tok = best_match['quote_start']
                end_tok = best_match['quote_end']
                
                context_start = max(0, start_tok - 20)
                context_end = min(len(token_bytes) - 1, end_tok + 20)
                
                context_words = b_tokens[(b_tokens['token_ID_within_document'] >= context_start) & 
                                         (b_tokens['token_ID_within_document'] <= context_end)]['word'].tolist()
                
                context_str = " ".join([str(w) for w in context_words])
                
                results.append({
                    'quote_id': q_id,
                    'mention_phrase': mention_phrase,
                    'quote_type': q_type,
                    'context': context_str
                })
            else:
                results.append({
                    'quote_id': q_id,
                    'mention_phrase': "",
                    'quote_type': QuoteType.IMPLICIT.value,
                    'context': "NO MATCH"
                })
                
    os.makedirs("results/EXP015", exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv("results/EXP015/sample_100_quotes.csv", index=False)
    
    print(res_df.head(10).to_string())
    print("Sample generated.")

if __name__ == "__main__":
    main()
