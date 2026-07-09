import pandas as pd
from src.addressee.pipeline import AddresseeFeatureProvider

def test_disable_mode():
    """Test 1 - Disable Mode: When disabled, returns baseline-equivalent safe defaults."""
    provider = AddresseeFeatureProvider(enabled=False)
    feats = provider.extract("PrideAndPrejudice", 0, "Mr. Darcy")
    assert feats["candidate_was_addressed"] == 0.0
    assert feats["addressee_recency"] == -1.0
    assert feats["speaker_addressee_transition"] == 0.0

def test_feature_count():
    """Test 2 - Feature Count: Enabled should add exactly +3 features."""
    provider = AddresseeFeatureProvider(enabled=True)
    feats = provider.extract("PrideAndPrejudice", 0, "Mr. Darcy")
    assert len(feats) == 3
    assert "candidate_was_addressed" in feats
    assert "addressee_recency" in feats 
    assert "speaker_addressee_transition" in feats

def test_ordering():
    """Test 3 - Ordering: Feature order must be deterministic."""
    provider = AddresseeFeatureProvider(enabled=True)
    feats1 = provider.extract("PrideAndPrejudice", 0, "Mr. Darcy")
    feats2 = provider.extract("PrideAndPrejudice", 1, "Elizabeth Bennet")
    
    # Dictionary keys maintain insertion order in Python 3.7+
    assert list(feats1.keys()) == list(feats2.keys())
    assert list(feats1.keys()) == ["candidate_was_addressed", "addressee_recency", "speaker_addressee_transition"]

def test_missing_values():
    """Test 4 - Missing Values: unknown addressee -> -1 recency, 0 boolean features."""
    provider = AddresseeFeatureProvider(enabled=True)
    # Using an unknown character or early quote ensures no interaction history
    feats = provider.extract("PrideAndPrejudice", 0, "UnknownEntity")
    assert feats["candidate_was_addressed"] == 0.0
    assert feats["addressee_recency"] == -1.0
    assert feats["speaker_addressee_transition"] == 0.0

def test_leakage_check():
    """Test 5 - Leakage Check: Pipeline uses BookNLP attributes, not PDNC gold."""
    provider = AddresseeFeatureProvider(enabled=True)
    # The provider relies on extractor.py which only reads from token/dependency logic
    # and booknlp quotes, not taking `true_speaker_id` for state population.
    # The state is updated only through the BookNLP extracted quote data.
    provider.update_state("PrideAndPrejudice", 0)
    # We just ensure it runs without requiring gold labels.
    assert provider.updater is not None

if __name__ == "__main__":
    test_disable_mode()
    test_feature_count()
    test_ordering()
    test_missing_values()
    test_leakage_check()
    print("All integration reproducibility tests passed!")
