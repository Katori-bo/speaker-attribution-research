import pytest
from src.style.state import CharacterStyleState

def test_style_state_updates():
    state = CharacterStyleState()
    
    # Update with valid speaker and text
    state.update("Alice", "Hello world")
    assert "Alice" in state.state.fingerprints
    fp = state.state.fingerprints["Alice"]
    assert fp.quotes_seen == 1
    assert fp.texts == ["Hello world"]
    assert fp.total_tokens == 2
    
    # Update with invalid speakers
    state.update(None, "hello")
    state.update("Unknown", "hello")
    state.update("nan", "hello")
    state.update("Alice", "")
    
    assert len(state.state.fingerprints) == 1
    assert fp.quotes_seen == 1
    assert fp.total_tokens == 2
