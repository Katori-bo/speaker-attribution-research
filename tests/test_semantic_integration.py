import os
import ast
import pandas as pd
from src.coreference.parser import BookNLPParser
from src.coreference.alignment import AlignmentLayer
from src.coreference.mapping import MentionToEntityMapper
from src.coreference.features import extract_coreference_features, NO_COREF_DISTANCE

def test_semantic_provider_determinism():
    """
    Integration test: verifies that given identical inputs, the semantic 
    provider produces exactly the same feature vector on repeated runs.
    """
    novel = "PrideAndPrejudice"
    out_dir = "data/raw/pdnc/booknlp_out"
    
    # We only run this if the BookNLP output exists (to avoid failing in CI without data)
    entities_path = os.path.join(out_dir, f"{novel}.entities")
    tokens_path = os.path.join(out_dir, f"{novel}.tokens")
    book_path = os.path.join(out_dir, f"{novel}.book")
    
    if not os.path.exists(entities_path) or not os.path.exists(tokens_path) or not os.path.exists(book_path):
        return
        
    parser = BookNLPParser()
    entities = parser.parse_entities(entities_path)
    aliases = parser.parse_book_aliases(book_path)
    tokens_df = pd.read_csv(tokens_path, sep='\t')
    alignment_layer = AlignmentLayer(tokens_df)
    mapper = MentionToEntityMapper(entities, aliases)
    
    # Pick an arbitrary quote span and candidate
    quote_spans = [[357, 442]]  # Just some byte span from P&P
    candidate_str = "Elizabeth"
    
    def extract_features():
        token_ids = alignment_layer.map_quote_byte_spans_to_tokens(quote_spans)
        quote_span = (token_ids[0], token_ids[-1])
        chain_id = mapper.resolve_string_to_chain_id(candidate_str)
        if chain_id is None:
            return {
                "candidate_in_quote_chain": False,
                "nearest_coref_dist": NO_COREF_DISTANCE,
                "recent_mention_count": 0,
                "chain_recency": NO_COREF_DISTANCE
            }
        return extract_coreference_features(chain_id, quote_span, entities)

    # Run 1
    run1_features = extract_features()
    
    # Run 2
    run2_features = extract_features()
    
    # Assert Exact Match
    assert run1_features == run2_features
    assert "candidate_in_quote_chain" in run1_features
    assert "nearest_coref_dist" in run1_features
    assert "recent_mention_count" in run1_features
    assert "chain_recency" in run1_features
