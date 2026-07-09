from typing import Dict, Any, List
from src.discourse.discourse_state import MinimalDiscourseState
from src.baseline.attribution_rules import rule_explicit_attribution, rule_dialogue_alternation

class FeatureExtractor:
    """
    Extracts fixed-length feature vectors for a (quote, candidate) pair
    from the discourse state and quote text.
    Organizes features into families for clean ablation studies.
    """
    
    def extract(self, quote: Dict[str, Any], candidate: str, state: MinimalDiscourseState) -> Dict[str, float]:
        """
        Extracts features and organizes them into explicit families.
        Returns a flat dictionary of features to be used by classical ML models.
        """
        features = {}
        
        quote_text = quote.get('quote_text', '')
        context_text = quote.get('context_text', '')
        
        # --- Lexical Features ---
        features['lexical_quote_length_chars'] = float(len(quote_text))
        features['lexical_quote_length_tokens'] = float(len(quote_text.split()))
        features['lexical_has_question_mark'] = 1.0 if '?' in quote_text else 0.0
        features['lexical_has_exclamation'] = 1.0 if '!' in quote_text else 0.0

        # --- Candidate Features ---
        features['candidate_is_explicit_mention'] = 1.0 if candidate.lower() in context_text.lower() else 0.0
        features['candidate_is_last_speaker'] = 1.0 if candidate == state.last_speaker else 0.0
        features['candidate_is_previous_speaker'] = 1.0 if candidate == state.previous_speaker else 0.0
        features['candidate_is_recent_mention'] = 1.0 if candidate in state.recent_mentions else 0.0
        
        # --- Discourse Features ---
        features['discourse_dialogue_position'] = float(state.dialogue_position)
        features['discourse_context_length'] = float(len(context_text))

        # --- Conversation Features ---
        features['conversation_turn_index'] = float(state.dialogue_position)
        features['conversation_length'] = float(state.conversation_length)
        features['conversation_speaker_change'] = 1.0 if (state.last_speaker and state.previous_speaker and state.last_speaker != state.previous_speaker) else 0.0
        
        # --- Symbolic Features ---
        explicit_fired, explicit_cand = rule_explicit_attribution(quote_text, context_text, state, [])
        features['symbolic_explicit_rule_fired'] = 1.0 if (explicit_fired and explicit_cand == candidate) else 0.0
        
        alt_fired, alt_cand = rule_dialogue_alternation(quote_text, context_text, state, [])
        features['symbolic_alternation_rule_fired'] = 1.0 if (alt_fired and alt_cand == candidate) else 0.0
        
        return features
