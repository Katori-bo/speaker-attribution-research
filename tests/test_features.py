import pytest
import hashlib
from copy import deepcopy
from src.features.extractor import FeatureExtractor
from src.discourse.discourse_state import MinimalDiscourseState

def test_feature_reproducibility():
    extractor = FeatureExtractor()
    state = MinimalDiscourseState()
    
    quote = {'quote_text': 'Hello world!', 'context_text': 'he said'}
    candidate = 'John'
    
    # Run 1
    state1 = deepcopy(state)
    features1 = extractor.extract(quote, candidate, state1)
    hash1 = hashlib.md5(str(sorted(features1.items())).encode()).hexdigest()
    
    # Run 2
    state2 = deepcopy(state)
    features2 = extractor.extract(quote, candidate, state2)
    hash2 = hashlib.md5(str(sorted(features2.items())).encode()).hexdigest()
    
    assert hash1 == hash2, "Feature generation is not deterministic"

def test_feature_leakage():
    # Verify that feature generation never accesses future state
    extractor = FeatureExtractor()
    state = MinimalDiscourseState()
    
    # State setup (past events only)
    state.last_speaker = 'Alice'
    state.previous_speaker = 'Bob'
    state.recent_mentions = ['Alice', 'Bob', 'Charlie']
    
    quote = {'quote_text': 'How are you?', 'context_text': 'Alice asked Bob'}
    candidate = 'Charlie'
    
    features = extractor.extract(quote, candidate, state)
    
    # The extractor should only use 'state' and 'quote'. 
    # It should not have access to any future speakers or future quotes.
    # We can check that the extracted features only depend on the current state.
    assert features['candidate_is_last_speaker'] == 0.0
    assert features['candidate_is_previous_speaker'] == 0.0
    assert features['candidate_is_recent_mention'] == 1.0
    
    # If the state gets updated with future information, it's the caller's fault,
    # but the extractor itself doesn't pull future data.
    
def test_symbolic_boolean_flags():
    extractor = FeatureExtractor()
    state = MinimalDiscourseState()
    
    quote = {'quote_text': 'I am here.', 'context_text': 'John said'}
    candidate = 'John'
    
    # Add John to candidates so explicit rule can fire
    state.current_candidates = {'John'}
    
    features = extractor.extract(quote, candidate, state)
    
    assert features['symbolic_explicit_rule_fired'] == 1.0
    assert features['symbolic_alternation_rule_fired'] == 0.0
