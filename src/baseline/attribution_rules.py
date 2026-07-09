from typing import Optional, Tuple
from src.discourse.discourse_state import MinimalDiscourseState

def rule_explicit_attribution(quote_text: str, context_text: str, state: MinimalDiscourseState, explicit_mentions: list) -> Tuple[bool, Optional[str]]:
    """
    If the context text explicitly names a candidate near a speaking verb.
    Applicable if there is context_text.
    Fires if a candidate matches.
    """
    if not context_text:
        return False, None
        
    for candidate in state.current_candidates:
        if candidate.lower() in context_text.lower():
            return True, candidate
            
    return True, None

def rule_dialogue_alternation(quote_text: str, context_text: str, state: MinimalDiscourseState, explicit_mentions: list) -> Tuple[bool, Optional[str]]:
    """
    A-B-A-B conversation structure.
    Applicable if there is a previous_speaker and a last_speaker.
    Fires if previous_speaker != last_speaker and previous is in candidates.
    """
    if state.previous_speaker and state.last_speaker:
        if state.previous_speaker != state.last_speaker:
            if state.previous_speaker in state.current_candidates:
                return True, state.previous_speaker
        return True, None
    return False, None

def rule_previous_speaker(quote_text: str, context_text: str, state: MinimalDiscourseState, explicit_mentions: list) -> Tuple[bool, Optional[str]]:
    """
    Continued speech (A-A). 
    Applicable if there is a last_speaker.
    Fires if last_speaker is in candidates.
    """
    if state.last_speaker:
        if state.last_speaker in state.current_candidates:
            return True, state.last_speaker
        return True, None
    return False, None

def rule_nearest_mention(quote_text: str, context_text: str, state: MinimalDiscourseState, explicit_mentions: list) -> Tuple[bool, Optional[str]]:
    """
    Fallback to the most recently mentioned character.
    Applicable if there are explicit mentions or recent mentions.
    Fires if one is in candidates.
    """
    if explicit_mentions or state.recent_mentions:
        if explicit_mentions:
            for mention in reversed(explicit_mentions):
                if mention in state.current_candidates:
                    return True, mention
                    
        if state.recent_mentions:
            for mention in reversed(state.recent_mentions):
                if mention in state.current_candidates:
                    return True, mention
        return True, None
    return False, None

ATTRIBUTION_RULES = [
    ("Explicit Attribution", rule_explicit_attribution),
    ("Dialogue Alternation", rule_dialogue_alternation),
    ("Previous Speaker", rule_previous_speaker),
    ("Nearest Mention", rule_nearest_mention)
]
