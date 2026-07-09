import os
import ast
import pandas as pd
from typing import Dict, Any
from src.coreference.parser import BookNLPParser
from src.coreference.alignment import AlignmentLayer
from src.coreference.mapping import MentionToEntityMapper
from src.coreference.features import extract_coreference_features, NO_COREF_DISTANCE

class SemanticFeatureProvider:
    """
    Integrates the semantic feature extractors into the ranking pipeline.
    Loads BookNLP representations on demand per novel and augments candidates
    with coreference features.
    """
    def __init__(self, booknlp_out_dir: str = "data/raw/pdnc/booknlp_out"):
        self.booknlp_out_dir = booknlp_out_dir
        self.parser = BookNLPParser()
        
        # Caches to avoid reloading for every quote
        self.current_novel = None
        self.entities = None
        self.alignment_layer = None
        self.mapper = None
        
    def _load_novel(self, novel: str):
        if self.current_novel == novel:
            return
            
        novel_dir = os.path.join(self.booknlp_out_dir, novel)
        entities_path = os.path.join(novel_dir, f"{novel}.entities")
        tokens_path = os.path.join(novel_dir, f"{novel}.tokens")
        book_path = os.path.join(novel_dir, f"{novel}.book")
        
        if not os.path.exists(entities_path) or not os.path.exists(tokens_path) or not os.path.exists(book_path):
            raise FileNotFoundError(f"BookNLP outputs missing for novel {novel}")
            
        self.entities = self.parser.parse_entities(entities_path)
        aliases = self.parser.parse_book_aliases(book_path)
        tokens_df = pd.read_csv(tokens_path, sep='\t')
        self.alignment_layer = AlignmentLayer(tokens_df)
        self.mapper = MentionToEntityMapper(self.entities, aliases)
        self.current_novel = novel

    def augment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augments a candidate features DataFrame with the 4 semantic features.
        The input DataFrame must contain: 'novel', 'quote_byte_spans' (or equivalent), and 'candidate'.
        """
        # Ensure we don't mutate original
        df = df.copy()
        
        features_list = []
        
        for idx, row in df.iterrows():
            novel = row['novel']
            candidate_str = row['candidate']
            
            # Use quoteByteSpans if available, else fallback to quote_start_byte/quote_end_byte
            spans = []
            if 'quoteByteSpans' in row and pd.notna(row['quoteByteSpans']):
                spans_str = row['quoteByteSpans']
                spans = ast.literal_eval(spans_str)
                if len(spans) > 0 and isinstance(spans[0], int):
                    spans = [spans]
            elif 'quote_start_byte' in row and 'quote_end_byte' in row:
                spans = [[row['quote_start_byte'], row['quote_end_byte']]]
                
            try:
                self._load_novel(novel)
                token_ids = self.alignment_layer.map_quote_byte_spans_to_tokens(spans)
                if not token_ids:
                    raise ValueError("No tokens aligned")
                    
                quote_span = (token_ids[0], token_ids[-1])
                chain_id = self.mapper.resolve_string_to_chain_id(candidate_str)
                
                if chain_id is None:
                    # Fallback for unmapped candidates
                    feats = {
                        "candidate_in_quote_chain": 0.0,
                        "nearest_coref_dist": float(NO_COREF_DISTANCE),
                        "recent_mention_count": 0.0,
                        "chain_recency": float(NO_COREF_DISTANCE)
                    }
                else:
                    raw_feats = extract_coreference_features(chain_id, quote_span, self.entities)
                    feats = {
                        "candidate_in_quote_chain": 1.0 if raw_feats["candidate_in_quote_chain"] else 0.0,
                        "nearest_coref_dist": float(raw_feats["nearest_coref_dist"]),
                        "recent_mention_count": float(raw_feats["recent_mention_count"]),
                        "chain_recency": float(raw_feats["chain_recency"])
                    }
            except Exception:
                # If anything fails (e.g., missing file, alignment failure), fallback gracefully
                feats = {
                    "candidate_in_quote_chain": 0.0,
                    "nearest_coref_dist": float(NO_COREF_DISTANCE),
                    "recent_mention_count": 0.0,
                    "chain_recency": float(NO_COREF_DISTANCE)
                }
                
            features_list.append(feats)
            
        new_feats_df = pd.DataFrame(features_list, index=df.index)
        
        # Merge back
        for col in new_feats_df.columns:
            df[col] = new_feats_df[col]
            
        return df
