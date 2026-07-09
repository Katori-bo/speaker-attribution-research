import pytest
from src.style.state import CharacterStyleState
from src.style.features import compute_similarity_scores

def test_lexical_similarity():
    state = CharacterStyleState()
    
    # Alice has 5 quotes (reaches min_quotes default of 5)
    for _ in range(5):
        state.update("Alice", "I hate tea and cookies")
        
    # Bob has 5 quotes
    for _ in range(5):
        state.update("Bob", "I love coffee and cake")
        
    # Charlie has 2 quotes (below min_quotes)
    for _ in range(2):
        state.update("Charlie", "I hate tea and cookies")
        
    candidates = ["Alice", "Bob", "Charlie", "Dave"]
    
    # Quote similar to Alice
    scores = compute_similarity_scores("cookies and tea", candidates, state, min_quotes=5)
    assert scores["Alice"] > scores["Bob"]
    assert scores["Charlie"] == 0.0  # below min_quotes
    assert scores["Dave"] == 0.0     # not in state
    
    # Quote similar to Bob
    scores = compute_similarity_scores("coffee and cake", candidates, state, min_quotes=5)
    assert scores["Bob"] > scores["Alice"]
    
    # Verify min_quotes threshold works when lowered
    scores_min_2 = compute_similarity_scores("cookies and tea", candidates, state, min_quotes=2)
    assert scores_min_2["Charlie"] > 0.0
