import os
import ast
import pandas as pd
import numpy as np
from src.coreference.parser import BookNLPParser
from src.coreference.alignment import AlignmentLayer
from src.coreference.features import extract_coreference_features, NO_COREF_DISTANCE

def generate_feature_statistics():
    novel = "PrideAndPrejudice"
    out_dir = "data/raw/pdnc/booknlp_out"
    pdnc_quotes = f"data/raw/pdnc/data/{novel}/quotation_info.csv"
    
    # Load Representation
    parser = BookNLPParser()
    entities = parser.parse_entities(os.path.join(out_dir, f"{novel}.entities"))
    tokens_df = pd.read_csv(os.path.join(out_dir, f"{novel}.tokens"), sep='\t')
    alignment_layer = AlignmentLayer(tokens_df)
    pdnc_df = pd.read_csv(pdnc_quotes)
    
    # Pick Top 5 Candidates (by mention count) to simulate candidate scoring
    top_candidates = sorted(entities.values(), key=lambda e: len(e.mentions), reverse=True)[:5]
    candidate_ids = [c.chain_id for c in top_candidates]
    
    features_data = []
    
    # Extract features for first 100 aligned quotes
    quotes_processed = 0
    for idx, row in pdnc_df.iterrows():
        if quotes_processed >= 100:
            break
            
        spans_str = row['quoteByteSpans']
        if pd.isna(spans_str):
            continue
        try:
            spans = ast.literal_eval(spans_str)
            if len(spans) > 0 and isinstance(spans[0], int):
                spans = [spans]
            token_ids = alignment_layer.map_quote_byte_spans_to_tokens(spans)
            if not token_ids:
                continue
                
            quote_span = (token_ids[0], token_ids[-1])
            
            # Generate features for each candidate for this quote
            for cid in candidate_ids:
                feats = extract_coreference_features(cid, quote_span, entities)
                feats['candidate_id'] = cid
                feats['quote_idx'] = quotes_processed
                features_data.append(feats)
                
            quotes_processed += 1
        except Exception:
            continue
            
    df = pd.DataFrame(features_data)
    
    print("=== Coverage and Missing Value Audit ===")
    print(f"Total Candidate-Quote pairs evaluated: {len(df)}")
    
    for col in ['candidate_in_quote_chain', 'nearest_coref_dist', 'recent_mention_count', 'chain_recency']:
        missing_count = (df[col] == NO_COREF_DISTANCE).sum() if col in ['nearest_coref_dist', 'chain_recency'] else 0
        coverage = ((len(df) - missing_count) / len(df)) * 100
        print(f"Feature: {col}")
        print(f"  Coverage: {coverage:.1f}%")
        print(f"  Missing (Encoded as {NO_COREF_DISTANCE}): {missing_count}")
        
    print("\n=== Distribution Statistics ===")
    for col in ['candidate_in_quote_chain', 'nearest_coref_dist', 'recent_mention_count', 'chain_recency']:
        if col == 'candidate_in_quote_chain':
            print(f"{col}: {df[col].mean():.3f} mean (True proportion)")
        else:
            # Exclude missing values for dist stats
            valid_df = df[df[col] != NO_COREF_DISTANCE]
            if len(valid_df) > 0:
                print(f"{col}: Mean {valid_df[col].mean():.2f}, Variance {valid_df[col].var():.2f}, Max {valid_df[col].max()}")
            else:
                print(f"{col}: No valid data")
                
    print("\n=== Feature Correlation Audit ===")
    # Calculate correlation matrix (using spearman to handle boolean/ordinal well)
    cols = ['candidate_in_quote_chain', 'nearest_coref_dist', 'recent_mention_count', 'chain_recency']
    corr = df[cols].corr(method='spearman')
    print(corr.round(3).to_string())
    
if __name__ == "__main__":
    generate_feature_statistics()
