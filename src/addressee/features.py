from typing import Dict, Any
from .schemas import InteractionState

def extract_addressee_features(
    candidate_id: int,
    current_quote_id: int,
    interaction_state: InteractionState
) -> Dict[str, Any]:
    
    # Feature 1: candidate_was_addressed
    # True if the candidate was the addressee of the most recent valid interaction.
    candidate_was_addressed = False
    if interaction_state.last_interaction and interaction_state.last_interaction.addressee_id == candidate_id:
        candidate_was_addressed = True
        
    # Feature 2: addressee_recency
    # Number of dialogue turns since this candidate was last addressed.
    addressee_recency = -1
    last_addressed = interaction_state.last_addressed_at.get(candidate_id)
    if last_addressed is not None:
        addressee_recency = current_quote_id - last_addressed
        
    # Feature 3: speaker_addressee_transition
    # Has the previous speaker recently addressed this candidate?
    # We check if there's a record of the previous speaker transitioning to the candidate.
    speaker_addressee_transition = False
    if interaction_state.last_interaction and interaction_state.last_interaction.speaker_id != -1:
        prev_speaker = interaction_state.last_interaction.speaker_id
        
        for transition in interaction_state.speaker_transition_history:
            if transition["from"] == prev_speaker and transition["to"] == candidate_id:
                speaker_addressee_transition = True
                break
                
    return {
        "candidate_was_addressed": candidate_was_addressed,
        "addressee_recency": addressee_recency,
        "speaker_addressee_transition": speaker_addressee_transition
    }
