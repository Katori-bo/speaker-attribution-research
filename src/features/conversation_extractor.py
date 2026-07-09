from typing import Dict, Any
from src.discourse.conversation_state import ConversationStateModule

class ConversationFeatureExtractor:
    """
    Extracts features from the ConversationStateModule for a specific candidate.
    """
    def extract(self, quote: Dict[str, Any], candidate: str, state: ConversationStateModule) -> Dict[str, float]:
        features = {}
        
        # 1. Active Conversation ID (Global)
        features['conv_active_id'] = float(state.active_conversation_id)
        
        # 2. Interruption Distance (Global, Physical Bytes)
        start_byte = quote.get("quote_start_byte", -1)
        features['conv_interruption_distance'] = float(state.get_interruption_distance(start_byte))
        
        # 3. Participant Stack Presence and Depth (Candidate-Specific)
        # Depth 1 = most recent, Depth 2 = second most recent, etc.
        try:
            # We reverse the stack conceptually: index 0 is oldest, index -1 is newest
            # We want depth 1 to be the newest speaker (top of stack)
            idx = state.participant_stack[::-1].index(candidate)
            features['candidate_in_participant_stack'] = 1.0
            features['candidate_stack_depth'] = float(idx + 1)
        except ValueError:
            features['candidate_in_participant_stack'] = 0.0
            features['candidate_stack_depth'] = 0.0  # Or a large penalty number

        return features
