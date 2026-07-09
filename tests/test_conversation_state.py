import pytest
from src.discourse.conversation_state import ConversationStateModule
from src.features.conversation_extractor import ConversationFeatureExtractor

def test_conversation_initialization():
    state = ConversationStateModule("scene_1")
    assert state.active_conversation_id == 0
    assert len(state.participant_stack) == 0
    assert state.quotes_since_last_turn == 0

def test_participant_stack_ordering():
    state = ConversationStateModule("scene_1")
    
    # Alice speaks
    state.update({"quote_text": "Hello"}, "Alice")
    assert state.participant_stack == ["Alice"]
    
    # Bob speaks
    state.update({"quote_text": "Hi Alice"}, "Bob")
    assert state.participant_stack == ["Alice", "Bob"]
    
    # Alice speaks again (should move to top)
    state.update({"quote_text": "How are you?"}, "Alice")
    assert state.participant_stack == ["Bob", "Alice"]
    
def test_interruption_distance():
    state = ConversationStateModule("scene_1")
    
    state.update({}, "Alice")
    assert state.quotes_since_last_turn == 1
    
    state.update({}, "Bob")
    assert state.quotes_since_last_turn == 2

def test_feature_extraction():
    state = ConversationStateModule("scene_1")
    extractor = ConversationFeatureExtractor()
    
    state.update({}, "Alice")
    state.update({}, "Bob")
    state.update({}, "Charlie")
    
    # Stack is [Alice, Bob, Charlie]
    
    features = extractor.extract({}, "Charlie", state)
    assert features["candidate_in_participant_stack"] == 1.0
    assert features["candidate_stack_depth"] == 1.0  # Most recent
    
    features = extractor.extract({}, "Bob", state)
    assert features["candidate_stack_depth"] == 2.0  # Second most recent
    
    features = extractor.extract({}, "Alice", state)
    assert features["candidate_stack_depth"] == 3.0
    
    features = extractor.extract({}, "David", state)
    assert features["candidate_in_participant_stack"] == 0.0
    
def test_reset_on_scene_boundary():
    state = ConversationStateModule("scene_1")
    state.update({}, "Alice")
    
    state.reset("scene_2")
    assert state.scene_id == "scene_2"
    assert state.active_conversation_id == 1
    assert len(state.participant_stack) == 0
    assert state.quotes_since_last_turn == 0
