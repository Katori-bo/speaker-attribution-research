import pytest
import pandas as pd
from src.coreference.validation import RepresentationValidator
from src.coreference.schemas import CanonicalEntity, Mention

def test_valid_chain():
    e = CanonicalEntity(chain_id=1, mentions=[
        Mention(10, 15, "Darcy", "PROP", "PER")
    ])
    errors = RepresentationValidator.validate_chain_integrity({1: e})
    assert not errors

def test_broken_chain_id():
    e = CanonicalEntity(chain_id=2, mentions=[])
    errors = RepresentationValidator.validate_chain_integrity({1: e})
    assert len(errors) == 1
    assert "Chain ID mismatch" in errors[0]

def test_duplicate_mention():
    m = Mention(10, 15, "Darcy", "PROP", "PER")
    e = CanonicalEntity(chain_id=1, mentions=[m, m])
    errors = RepresentationValidator.validate_chain_integrity({1: e})
    assert len(errors) == 1
    assert "Duplicate mention" in errors[0]

def test_invalid_offsets():
    m = Mention(20, 10, "Darcy", "PROP", "PER")
    e = CanonicalEntity(chain_id=1, mentions=[m])
    errors = RepresentationValidator.validate_mention_ordering({1: e})
    assert len(errors) == 1
    assert "Invalid offsets" in errors[0]

def test_empty_chain():
    e = CanonicalEntity(chain_id=1, mentions=[])
    errors = RepresentationValidator.validate_canonical_mapping({1: e})
    assert len(errors) == 1
    assert "Empty chain" in errors[0]
