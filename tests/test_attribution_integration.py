import pytest
from src.attribution.pipeline import AttributionFeatureProvider
from src.coreference.schemas import CanonicalEntity, Mention
from src.coreference.mapping import MentionToEntityMapper

def setup_mapper():
    # Setup dummy entities
    # Darcy = chain 1
    # Elizabeth = chain 2
    
    darcy_entity = CanonicalEntity(chain_id=1, mentions=[])
    elizabeth_entity = CanonicalEntity(chain_id=2, mentions=[])
    
    entities = {
        1: darcy_entity,
        2: elizabeth_entity
    }
    
    aliases = {
        1: {"Darcy", "Mr. Darcy", "Fitzwilliam Darcy"},
        2: {"Elizabeth", "Elizabeth Bennet", "Miss Bennet", "Lizzy"}
    }
    
    return MentionToEntityMapper(entities, aliases)

def get_text_and_bounds(text: str) -> tuple[str, int, int]:
    """Helper to extract quote bounds from a test string."""
    start = text.find('"')
    end = text.rfind('"') + 1
    return text, start, end

def test_baseline_reproduction():
    """Test 1: Disabled provider should return 0.0 feature."""
    mapper = setup_mapper()
    provider = AttributionFeatureProvider(mapper, enabled=False)
    
    text, start, end = get_text_and_bounds('"Hello," said Darcy.')
    
    features = provider.get_features(candidate_chain_id=1, quote_id="q1", quote_start=start, quote_end=end, content=text)
    
    assert "candidate_is_attributed_speaker" in features
    assert features["candidate_is_attributed_speaker"] == 0.0

def test_feature_count():
    """Test 2: Exactly +1 feature."""
    mapper = setup_mapper()
    provider = AttributionFeatureProvider(mapper, enabled=True)
    
    text, start, end = get_text_and_bounds('"Hello," said Darcy.')
    
    features = provider.get_features(candidate_chain_id=1, quote_id="q1", quote_start=start, quote_end=end, content=text)
    
    assert len(features) == 1
    assert "candidate_is_attributed_speaker" in features

def test_known_positive():
    """Test 3: Known positive candidate matches."""
    mapper = setup_mapper()
    provider = AttributionFeatureProvider(mapper, enabled=True)
    
    text, start, end = get_text_and_bounds('"Hello," said Darcy.')
    
    # Darcy (1) is the speaker
    f_darcy = provider.get_features(candidate_chain_id=1, quote_id="q1", quote_start=start, quote_end=end, content=text)
    assert f_darcy["candidate_is_attributed_speaker"] == 1.0
    
    # Elizabeth (2) is not
    f_elizabeth = provider.get_features(candidate_chain_id=2, quote_id="q1", quote_start=start, quote_end=end, content=text)
    assert f_elizabeth["candidate_is_attributed_speaker"] == 0.0

def test_object_rejection():
    """Test 4: Object rejection."""
    mapper = setup_mapper()
    provider = AttributionFeatureProvider(mapper, enabled=True)
    
    text, start, end = get_text_and_bounds('"Hello," said Darcy to Elizabeth.')
    
    # Darcy is subject (1)
    f_darcy = provider.get_features(candidate_chain_id=1, quote_id="q2", quote_start=start, quote_end=end, content=text)
    assert f_darcy["candidate_is_attributed_speaker"] == 1.0
    
    # Elizabeth is object (2)
    f_elizabeth = provider.get_features(candidate_chain_id=2, quote_id="q2", quote_start=start, quote_end=end, content=text)
    assert f_elizabeth["candidate_is_attributed_speaker"] == 0.0

def test_pronoun_rejection():
    """Test 5: Pronoun rejection."""
    mapper = setup_mapper()
    provider = AttributionFeatureProvider(mapper, enabled=True)
    
    text, start, end = get_text_and_bounds('"Hello," said he.')
    
    f_darcy = provider.get_features(candidate_chain_id=1, quote_id="q3", quote_start=start, quote_end=end, content=text)
    assert f_darcy["candidate_is_attributed_speaker"] == 0.0

def test_nominal_rejection():
    """Test 6: No nominal extraction."""
    mapper = setup_mapper()
    provider = AttributionFeatureProvider(mapper, enabled=True)
    
    text, start, end = get_text_and_bounds('"Come here," said the old man.')
    
    f_darcy = provider.get_features(candidate_chain_id=1, quote_id="q4", quote_start=start, quote_end=end, content=text)
    assert f_darcy["candidate_is_attributed_speaker"] == 0.0
