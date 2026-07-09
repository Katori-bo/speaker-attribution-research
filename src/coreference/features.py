from typing import Dict, Tuple, List
from .schemas import CanonicalEntity, Quote

NO_COREF_DISTANCE = -1

def _candidate_in_quote_chain(candidate_chain_id: int, quote_start: int, quote_end: int, entities: Dict[int, CanonicalEntity]) -> bool:
    """
    True iff any mention overlapping the quote span belongs to the candidate's coreference chain.
    """
    candidate_entity = entities.get(candidate_chain_id)
    if not candidate_entity:
        return False
        
    for mention in candidate_entity.mentions:
        if max(quote_start, mention.start_token) <= min(quote_end, mention.end_token):
            return True
    return False

def _nearest_coref_dist(candidate_chain_id: int, quote_start: int, quote_end: int, entities: Dict[int, CanonicalEntity]) -> int:
    """
    Absolute token distance from the quote's boundaries to the nearest coreferential mention of the candidate.
    Returns NO_COREF_DISTANCE if no mention exists outside the quote.
    """
    candidate_entity = entities.get(candidate_chain_id)
    if not candidate_entity:
        return NO_COREF_DISTANCE
        
    min_dist = float('inf')
    found = False
    
    for mention in candidate_entity.mentions:
        # Ignore mentions inside the quote (covered by candidate_in_quote_chain)
        if max(quote_start, mention.start_token) <= min(quote_end, mention.end_token):
            continue
            
        if mention.end_token < quote_start:
            dist = quote_start - mention.end_token
        else:
            dist = mention.start_token - quote_end
            
        if dist < min_dist:
            min_dist = dist
            found = True
            
    return int(min_dist) if found else NO_COREF_DISTANCE

def _recent_mention_count(candidate_chain_id: int, quote_start: int, context_window_tokens: int, entities: Dict[int, CanonicalEntity]) -> int:
    """
    Count of mentions for this candidate's chain in the last N tokens prior to the quote.
    """
    candidate_entity = entities.get(candidate_chain_id)
    if not candidate_entity:
        return 0
        
    window_start = max(0, quote_start - context_window_tokens)
    count = 0
    
    for mention in candidate_entity.mentions:
        # Must be strictly before the quote, and overlap the window
        if mention.end_token < quote_start and mention.end_token >= window_start:
            count += 1
            
    return count

def _chain_recency(candidate_chain_id: int, quote_start: int, entities: Dict[int, CanonicalEntity]) -> int:
    """
    Count only unique chains between the previous occurrence of the candidate chain and the quote boundary.
    Returns NO_COREF_DISTANCE if candidate chain never appeared before the quote.
    """
    candidate_entity = entities.get(candidate_chain_id)
    if not candidate_entity:
        return NO_COREF_DISTANCE
        
    # Find last occurrence of candidate before the quote
    last_candidate_end = -1
    for mention in candidate_entity.mentions:
        if mention.end_token < quote_start:
            if mention.end_token > last_candidate_end:
                last_candidate_end = mention.end_token
                
    if last_candidate_end == -1:
        return NO_COREF_DISTANCE
        
    # Count unique chains mentioned between `last_candidate_end` and `quote_start`
    unique_intervening_chains = set()
    for chain_id, entity in entities.items():
        if chain_id == candidate_chain_id:
            continue
        for mention in entity.mentions:
            # Overlaps the gap
            if mention.start_token > last_candidate_end and mention.end_token < quote_start:
                unique_intervening_chains.add(chain_id)
                break # We only need to know if it appeared at least once in the gap
                
    return len(unique_intervening_chains)

def extract_coreference_features(candidate_chain_id: int, quote_span: Tuple[int, int], entities: Dict[int, CanonicalEntity], context_window_tokens: int = 50) -> Dict[str, float]:
    """
    Extracts explicit coreference-based semantic features for a candidate relative to a quote.
    
    Returns a flat dictionary containing:
    - candidate_in_quote_chain (bool)
    - nearest_coref_dist (int)
    - recent_mention_count (int)
    - chain_recency (int)
    """
    quote_start, quote_end = quote_span
    
    return {
        "candidate_in_quote_chain": _candidate_in_quote_chain(candidate_chain_id, quote_start, quote_end, entities),
        "nearest_coref_dist": _nearest_coref_dist(candidate_chain_id, quote_start, quote_end, entities),
        "recent_mention_count": _recent_mention_count(candidate_chain_id, quote_start, context_window_tokens, entities),
        "chain_recency": _chain_recency(candidate_chain_id, quote_start, entities)
    }
