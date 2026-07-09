import pytest
import pandas as pd
from src.coreference.alignment import AlignmentLayer

def test_alignment_exact_match():
    tokens_df = pd.DataFrame({
        "token_ID_within_document": [0, 1, 2],
        "byte_onset": [0, 5, 10],
        "byte_offset": [4, 9, 14]
    })
    layer = AlignmentLayer(tokens_df)
    
    # Matches token 1
    tokens = layer.map_quote_byte_spans_to_tokens([[5, 8]])
    assert tokens == [1]

def test_alignment_span_multiple_tokens():
    tokens_df = pd.DataFrame({
        "token_ID_within_document": [0, 1, 2],
        "byte_onset": [0, 5, 10],
        "byte_offset": [4, 9, 14]
    })
    layer = AlignmentLayer(tokens_df)
    
    # Matches tokens 0, 1, 2
    tokens = layer.map_quote_byte_spans_to_tokens([[2, 12]])
    assert tokens == [0, 1, 2]

def test_alignment_discontinuous_spans():
    tokens_df = pd.DataFrame({
        "token_ID_within_document": [0, 1, 2, 3],
        "byte_onset": [0, 5, 10, 15],
        "byte_offset": [4, 9, 14, 19]
    })
    layer = AlignmentLayer(tokens_df)
    
    # Matches token 0 and 3
    tokens = layer.map_quote_byte_spans_to_tokens([[0, 3], [15, 18]])
    assert tokens == [0, 3]

def test_alignment_malformed_token_df():
    # Include a non-integer row to simulate tab parsing issues
    tokens_df = pd.DataFrame({
        "token_ID_within_document": [0, "PUNCT", 2],
        "byte_onset": [0, "PUNCT", 10],
        "byte_offset": [4, "PUNCT", 14]
    })
    layer = AlignmentLayer(tokens_df)
    
    # It should skip the malformed row and successfully parse others
    tokens = layer.map_quote_byte_spans_to_tokens([[0, 12]])
    assert tokens == [0, 2]
