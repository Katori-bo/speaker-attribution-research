import pytest
import pandas as pd
from pathlib import Path
from src.coreference.parser import BookNLPParser
from src.coreference.mapping import MentionToEntityMapper

@pytest.fixture
def parser():
    return BookNLPParser()

def test_malformed_entity_file(parser, tmp_path):
    # Missing required columns
    file_path = tmp_path / "malformed.entities"
    df = pd.DataFrame({"COREF": [1], "text": ["hello"]})
    df.to_csv(file_path, sep='\t', index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        parser.parse_entities(str(file_path))

def test_missing_entity_ids(parser, tmp_path):
    # COREF column has NaN
    file_path = tmp_path / "missing_coref.entities"
    df = pd.DataFrame({
        "COREF": [float('nan')],
        "start_token": [0],
        "end_token": [1],
        "prop": ["PRON"],
        "cat": ["PER"],
        "text": ["he"]
    })
    df.to_csv(file_path, sep='\t', index=False)

    with pytest.raises(ValueError, match="Missing entity ID"):
        parser.parse_entities(str(file_path))

def test_overlapping_mentions(parser, tmp_path):
    file_path = tmp_path / "overlap.entities"
    df = pd.DataFrame({
        "COREF": [1, 2],
        "start_token": [5, 6],
        "end_token": [10, 8],
        "prop": ["PROP", "NOM"],
        "cat": ["PER", "PER"],
        "text": ["Mr. Darcy", "Darcy"]
    })
    df.to_csv(file_path, sep='\t', index=False)

    entities = parser.parse_entities(str(file_path))
    assert len(entities) == 2
    assert len(entities[1].mentions) == 1
    assert len(entities[2].mentions) == 1

def test_invalid_offsets(parser, tmp_path):
    file_path = tmp_path / "invalid_offsets.entities"
    df = pd.DataFrame({
        "COREF": [1],
        "start_token": [10],
        "end_token": [5],  # end < start
        "prop": ["PRON"],
        "cat": ["PER"],
        "text": ["he"]
    })
    df.to_csv(file_path, sep='\t', index=False)

    with pytest.raises(ValueError, match="Invalid offsets: start_token"):
        parser.parse_entities(str(file_path))

def test_valid_mention_to_entity_mapping(parser, tmp_path):
    file_path = tmp_path / "valid.entities"
    df = pd.DataFrame({
        "COREF": [1, 1, 2],
        "start_token": [10, 20, 30],
        "end_token": [11, 21, 31],
        "prop": ["PROP", "PRON", "PROP"],
        "cat": ["PER", "PER", "PER"],
        "text": ["Darcy", "he", "Elizabeth"]
    })
    df.to_csv(file_path, sep='\t', index=False)

    entities = parser.parse_entities(str(file_path))
    assert len(entities) == 2
    assert len(entities[1].mentions) == 2
    assert len(entities[2].mentions) == 1

    mapper = MentionToEntityMapper(entities)
    
    # Resolve valid mention
    entity = mapper.resolve_mention(10, 11)
    assert entity is not None
    assert entity.chain_id == 1

    # Resolve invalid mention
    entity = mapper.resolve_mention(15, 16)
    assert entity is None

def test_missing_quote_alignment(parser, tmp_path):
    file_path = tmp_path / "missing_alignment.quotes"
    df = pd.DataFrame({
        "quote_start": [float('nan')],
        "quote_end": [5],
        "quote": ["Hello"],
        "char_id": [1]
    })
    df.to_csv(file_path, sep='\t', index=False)

    with pytest.raises(ValueError, match="Missing quote alignment"):
        parser.parse_quotes(str(file_path))

def test_invalid_quote_offsets(parser, tmp_path):
    file_path = tmp_path / "invalid_quote_offsets.quotes"
    df = pd.DataFrame({
        "quote_start": [10],
        "quote_end": [5],
        "quote": ["Hello"],
        "char_id": [1]
    })
    df.to_csv(file_path, sep='\t', index=False)

    with pytest.raises(ValueError, match="Invalid offsets for quote"):
        parser.parse_quotes(str(file_path))

def test_valid_quote_parsing(parser, tmp_path):
    file_path = tmp_path / "valid.quotes"
    df = pd.DataFrame({
        "quote_start": [10, 20],
        "quote_end": [15, 25],
        "quote": ["Hello", "World"],
        "char_id": [1, float('nan')]
    })
    df.to_csv(file_path, sep='\t', index=False)

    quotes = parser.parse_quotes(str(file_path))
    assert len(quotes) == 2
    assert quotes[0].start_token == 10
    assert quotes[0].speaker_chain_id == 1
    
    assert quotes[1].start_token == 20
    assert quotes[1].speaker_chain_id is None
