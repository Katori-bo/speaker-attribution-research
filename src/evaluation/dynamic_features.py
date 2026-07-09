from typing import Dict
from src.discourse.discourse_state import MinimalDiscourseState
from src.discourse.conversation_state import ConversationStateModule

def extract_dynamic_features(candidate: str, state: MinimalDiscourseState, conv_state: ConversationStateModule, current_quote_start_byte: int, style_scores: Dict[str, float] = None) -> Dict[str, float]:
    """
    Extracts ONLY the features that depend on discourse/conversation history.
    This allows us to cleanly overwrite these columns during autoregressive evaluation
    without perturbing any static features (like coref, attribution, or lexical properties).
    """
    features = {}
    
    # Discourse State Features
    features['candidate_is_last_speaker'] = 1.0 if candidate == state.last_speaker else 0.0
    features['candidate_is_previous_speaker'] = 1.0 if candidate == state.previous_speaker else 0.0
    features['candidate_is_recent_mention'] = 1.0 if candidate in state.recent_mentions else 0.0
    features['discourse_dialogue_position'] = float(state.dialogue_position)
    features['conversation_turn_index'] = float(state.dialogue_position)
    features['conversation_length'] = float(state.conversation_length)
    features['conversation_speaker_change'] = 1.0 if (state.last_speaker and state.previous_speaker and state.last_speaker != state.previous_speaker) else 0.0
    
    # Conversation State Features
    features['conv_active_id'] = float(conv_state.active_conversation_id)
    features['conv_interruption_distance'] = float(conv_state.get_interruption_distance(current_quote_start_byte))
    
    in_stack = candidate in conv_state.participant_stack
    features['candidate_in_participant_stack'] = 1.0 if in_stack else 0.0
    
    depth = 0.0
    if in_stack:
        # Depth is 1 for the most recent participant (end of list)
        idx = conv_state.participant_stack.index(candidate)
        depth = float(len(conv_state.participant_stack) - idx)
    features['candidate_stack_depth'] = depth
    
    # Character Lexical Fingerprint Similarity
    features['character_lexical_similarity'] = style_scores.get(candidate, 0.0) if style_scores is not None else 0.0
    
    return features
