from src.addressee.schemas import DialogueInteraction, ExtractionMethod
from src.addressee.state import InteractionStateUpdater

def test_state_update_ordering():
    updater = InteractionStateUpdater(history_limit=2)
    
    # Interaction 1: Speaker A to Unknown
    i1 = DialogueInteraction(1, 100, None, 0.0, ExtractionMethod.UNKNOWN)
    state = updater.update(i1)
    assert state.last_interaction == i1
    assert state.recent_addressees == []
    assert state.speaker_transition_history == []
    
    # Interaction 2: Speaker B to A
    i2 = DialogueInteraction(2, 200, 100, 0.9, ExtractionMethod.SPEECH_TAG_OBJECT)
    state = updater.update(i2)
    assert state.last_interaction == i2
    assert state.recent_addressees == [100]
    assert state.speaker_transition_history == [{"from": 100, "to": 200}]
    
    # Interaction 3: Speaker A to C
    i3 = DialogueInteraction(3, 100, 300, 0.6, ExtractionMethod.VOCATIVE)
    state = updater.update(i3)
    assert state.last_interaction == i3
    assert state.recent_addressees == [300, 100]
    assert state.speaker_transition_history == [{"from": 100, "to": 200}, {"from": 200, "to": 100}]
    
    # Interaction 4: Speaker C to Unknown
    i4 = DialogueInteraction(4, 300, None, 0.0, ExtractionMethod.UNKNOWN)
    state = updater.update(i4)
    # limit is 2
    assert state.recent_addressees == [300, 100]
    assert state.speaker_transition_history == [{"from": 200, "to": 100}, {"from": 100, "to": 300}]

def test_repeated_speakers():
    updater = InteractionStateUpdater(history_limit=5)
    
    # A to B
    updater.update(DialogueInteraction(1, 100, 200, 0.9, ExtractionMethod.SPEECH_TAG_OBJECT))
    # B to A
    updater.update(DialogueInteraction(2, 200, 100, 0.9, ExtractionMethod.SPEECH_TAG_OBJECT))
    # B to A again (e.g. multi-quote turn)
    updater.update(DialogueInteraction(3, 200, 100, 0.9, ExtractionMethod.SPEECH_TAG_OBJECT))
    
    # We should have A at the front of recent addressees, but only once
    assert updater.state.recent_addressees == [100, 200]
    
    assert updater.state.speaker_transition_history[-1] == {"from": 200, "to": 200}
    assert updater.state.speaker_transition_history[-2] == {"from": 100, "to": 200}

if __name__ == "__main__":
    test_state_update_ordering()
    test_repeated_speakers()
    print("All interaction state tests passed!")
