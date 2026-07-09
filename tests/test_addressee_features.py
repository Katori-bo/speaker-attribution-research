from src.addressee.schemas import DialogueInteraction, ExtractionMethod, InteractionState
from src.addressee.state import InteractionStateUpdater
from src.addressee.features import extract_addressee_features

def test_features_empty_state():
    state = InteractionState()
    features = extract_addressee_features(candidate_id=100, current_quote_id=1, interaction_state=state)
    
    assert features["candidate_was_addressed"] is False
    assert features["addressee_recency"] == -1
    assert features["speaker_addressee_transition"] is False

def test_candidate_was_addressed():
    updater = InteractionStateUpdater(history_limit=5)
    
    # Q1: 200 addresses 100
    interaction = DialogueInteraction(quote_id=1, speaker_id=200, addressee_id=100, confidence=0.9, extraction_method=ExtractionMethod.VOCATIVE)
    state = updater.update(interaction)
    
    # Evaluate for Q2, candidate 100
    features = extract_addressee_features(candidate_id=100, current_quote_id=2, interaction_state=state)
    assert features["candidate_was_addressed"] is True
    assert features["addressee_recency"] == 1
    
    # Evaluate for Q2, candidate 300
    features = extract_addressee_features(candidate_id=300, current_quote_id=2, interaction_state=state)
    assert features["candidate_was_addressed"] is False
    assert features["addressee_recency"] == -1

def test_speaker_addressee_transition():
    updater = InteractionStateUpdater(history_limit=5)
    
    # Q1: 100 speaks
    updater.update(DialogueInteraction(1, 100, None, 0.0, ExtractionMethod.UNKNOWN))
    # Q2: 200 speaks
    state = updater.update(DialogueInteraction(2, 200, None, 0.0, ExtractionMethod.UNKNOWN))
    
    # Now we are evaluating for Q3. The last speaker was 200.
    # We want to know if 200 transitioned to 100 in the past.
    # Q1 was 100, Q2 was 200. So transition is 100 -> 200.
    # Is there a transition from 200 to 100? No.
    features_100 = extract_addressee_features(candidate_id=100, current_quote_id=3, interaction_state=state)
    assert features_100["speaker_addressee_transition"] is False
    
    # Q3: 100 speaks (Transition 200 -> 100)
    state = updater.update(DialogueInteraction(3, 100, None, 0.0, ExtractionMethod.UNKNOWN))
    
    # Q4: 200 speaks (Transition 100 -> 200)
    state = updater.update(DialogueInteraction(4, 200, None, 0.0, ExtractionMethod.UNKNOWN))
    
    # Evaluate for Q5. Last speaker was 200.
    # Did 200 ever transition to 100? Yes, at Q3 (Q2 was 200, Q3 was 100).
    features_100 = extract_addressee_features(candidate_id=100, current_quote_id=5, interaction_state=state)
    assert features_100["speaker_addressee_transition"] is True
    
def test_unknown_addressee_handling():
    updater = InteractionStateUpdater(history_limit=5)
    
    # Q1: 200 addresses 100
    updater.update(DialogueInteraction(quote_id=1, speaker_id=200, addressee_id=100, confidence=0.9, extraction_method=ExtractionMethod.VOCATIVE))
    
    # Q2: 100 speaks, unknown addressee
    state = updater.update(DialogueInteraction(quote_id=2, speaker_id=100, addressee_id=None, confidence=0.0, extraction_method=ExtractionMethod.UNKNOWN))
    
    # Evaluate for Q3, candidate 100
    features = extract_addressee_features(candidate_id=100, current_quote_id=3, interaction_state=state)
    
    # 100 was addressed in Q1, so for Q3, recency = 3 - 1 = 2
    assert features["addressee_recency"] == 2
    # In Q2, the addressee was unknown, so 100 was NOT the last addressed.
    assert features["candidate_was_addressed"] is False

if __name__ == "__main__":
    test_features_empty_state()
    test_candidate_was_addressed()
    test_speaker_addressee_transition()
    test_unknown_addressee_handling()
    print("All feature tests passed!")
