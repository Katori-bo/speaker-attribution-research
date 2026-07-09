import pytest
from src.baseline.attribution_rules import rule_explicit_attribution, rule_dialogue_alternation, rule_nearest_mention
from src.discourse.discourse_state import MinimalDiscourseState

def test_explicit_attribution_rule():
    state = MinimalDiscourseState()
    state.current_candidates = {"Alice", "Bob"}
    
    context = '"I will go," Alice said.'
    pred, conf, rule = rule_explicit_attribution("", context, state)
    
    assert pred == "Alice"
    assert conf == 1.0
    assert rule == "Explicit Attribution"
    
def test_explicit_attribution_rule_fails_if_no_match():
    state = MinimalDiscourseState()
    state.current_candidates = {"Alice", "Bob"}
    
    context = '"I will go," she murmured.'
    pred, conf, rule = rule_explicit_attribution("", context, state)
    
    assert pred is None

def test_dialogue_alternation_rule():
    state = MinimalDiscourseState()
    state.current_candidates = {"Alice", "Bob"}
    state.previous_speaker = "Alice"
    state.last_speaker = "Bob"
    
    pred, conf, rule = rule_dialogue_alternation("", "", state)
    
    assert pred == "Alice"
    assert conf == 0.8
    assert rule == "Dialogue Alternation"
