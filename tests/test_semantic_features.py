import pytest
from src.coreference.features import (
    _candidate_in_quote_chain,
    _nearest_coref_dist,
    _recent_mention_count,
    _chain_recency,
    extract_coreference_features,
    NO_COREF_DISTANCE
)
from src.coreference.schemas import CanonicalEntity, Mention

def get_mock_entities():
    # Chain 1: Mentions at [10, 15], [30, 35], [50, 52]
    # Chain 2: Mentions at [20, 25], [40, 45]
    # Chain 3: Mentions at [100, 105]
    return {
        1: CanonicalEntity(1, [
            Mention(10, 15, "he", "PRON", "PER"),
            Mention(30, 35, "Mr. Darcy", "PROP", "PER"),
            Mention(50, 52, "Darcy", "PROP", "PER")
        ]),
        2: CanonicalEntity(2, [
            Mention(20, 25, "Elizabeth", "PROP", "PER"),
            Mention(40, 45, "she", "PRON", "PER")
        ]),
        3: CanonicalEntity(3, [
            Mention(100, 105, "Bingley", "PROP", "PER")
        ])
    }

def test_candidate_in_quote_chain():
    entities = get_mock_entities()
    
    # Quote from 28 to 38, overlaps with mention at [30, 35] of chain 1
    assert _candidate_in_quote_chain(1, 28, 38, entities) == True
    assert _candidate_in_quote_chain(2, 28, 38, entities) == False
    
    # Quote from 10 to 12, overlaps with [10, 15]
    assert _candidate_in_quote_chain(1, 10, 12, entities) == True

def test_nearest_coref_dist():
    entities = get_mock_entities()
    
    # Quote at 60-70. Chain 1 has mention at 50-52. Dist = 60 - 52 = 8
    assert _nearest_coref_dist(1, 60, 70, entities) == 8
    
    # Quote at 16-18. Chain 1 has mentions at 10-15 and 30-35. 
    # Dist to 10-15 is 1. Dist to 30-35 is 30 - 18 = 12. Min is 1.
    assert _nearest_coref_dist(1, 16, 18, entities) == 1
    
    # Chain 3 only has mention at 100-105. Quote is 60-70. Dist = 100 - 70 = 30.
    assert _nearest_coref_dist(3, 60, 70, entities) == 30
    
    # Unknown chain
    assert _nearest_coref_dist(99, 60, 70, entities) == NO_COREF_DISTANCE

def test_recent_mention_count():
    entities = get_mock_entities()
    
    # Quote at 60. Window 50 (looks back to 10).
    # Chain 1 mentions: 10-15, 30-35, 50-52 (all end strictly before 60, and >= 10).
    assert _recent_mention_count(1, 60, 50, entities) == 3
    
    # Quote at 40. Window 20 (looks back to 20).
    # Chain 1 mentions: 10-15 (misses window), 30-35 (in window).
    assert _recent_mention_count(1, 40, 20, entities) == 1
    
    # Quote at 100. Window 10 (looks back to 90). Chain 1 has no mentions in window.
    assert _recent_mention_count(1, 100, 10, entities) == 0

def test_chain_recency():
    entities = get_mock_entities()
    
    # Quote at 60.
    # Candidate Chain 1 last appeared at 50-52.
    # No other chains appeared between 52 and 60.
    assert _chain_recency(1, 60, entities) == 0
    
    # Quote at 46. Candidate Chain 1 last appeared at 30-35.
    # Between 35 and 46, Chain 2 appears at 40-45 (starts at 40 > 35, ends at 45 < 46).
    assert _chain_recency(1, 46, entities) == 1
    
    # Quote at 120. Candidate Chain 1 last appeared at 50-52.
    # Between 52 and 120, Chain 3 appeared (100-105).
    assert _chain_recency(1, 120, entities) == 1
    
    # Candidate Chain 3 before quote at 60 (has not appeared yet).
    assert _chain_recency(3, 60, entities) == NO_COREF_DISTANCE
