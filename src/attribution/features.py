from typing import Dict, Optional

def extract_attribution_features(candidate_chain_id: int, resolved_speaker_chain_id: Optional[int]) -> Dict[str, float]:
    """
    Extracts explicit attribution features.
    Strictly returns one feature: candidate_is_attributed_speaker
    """
    is_speaker = 1.0 if (resolved_speaker_chain_id is not None and candidate_chain_id == resolved_speaker_chain_id) else 0.0
    
    return {
        "candidate_is_attributed_speaker": is_speaker
    }
