import pytest
from src.discourse.discourse_state import MinimalDiscourseState

def test_state_updates_correctly():
    state = MinimalDiscourseState()
    
    assert state.last_speaker is None
    assert state.dialogue_position == 0
    
    state.update("Alice", ["Alice", "Bob"], {"Alice", "Bob"})
    
    assert state.last_speaker == "Alice"
    assert state.previous_speaker is None
    assert state.dialogue_position == 1
    
    state.update("Bob", ["Bob"], {"Alice", "Bob"})
    
    assert state.last_speaker == "Bob"
    assert state.previous_speaker == "Alice"
    assert state.dialogue_position == 2
    
def test_state_reset():
    state = MinimalDiscourseState()
    state.update("Alice", [], set())
    
    state.reset_conversation()
    
    assert state.last_speaker is None
    assert state.previous_speaker is None
    assert state.dialogue_position == 0
    assert state.conversation_length == 0
