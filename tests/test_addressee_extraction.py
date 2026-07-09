from src.addressee.extractor import AddresseeExtractor
from src.addressee.schemas import ExtractionMethod

def get_base_entities():
    return [
        {"COREF": "100", "start_token": "5", "end_token": "5"}, # Darcy
        {"COREF": "200", "start_token": "7", "end_token": "7"}, # Elizabeth
        {"COREF": "200", "start_token": "2", "end_token": "2"}, # Elizabeth inside quote
    ]

def test_extract_speech_tag_object():
    base_entities = get_base_entities()
    # "said Darcy to Elizabeth"
    tokens = [
        {"token_ID_within_document": 0, "sentence_ID": 0, "syntactic_head_ID": 0, "dependency_relation": "ROOT", "word": '"'},
        {"token_ID_within_document": 1, "sentence_ID": 0, "syntactic_head_ID": 0, "dependency_relation": "ROOT", "word": "Hello"},
        {"token_ID_within_document": 2, "sentence_ID": 0, "syntactic_head_ID": 0, "dependency_relation": "ROOT", "word": '"'},
        {"token_ID_within_document": 3, "sentence_ID": 0, "syntactic_head_ID": 0, "dependency_relation": "ROOT", "word": ","},
        {"token_ID_within_document": 4, "sentence_ID": 0, "syntactic_head_ID": 0, "dependency_relation": "ROOT", "word": "said"},
        {"token_ID_within_document": 5, "sentence_ID": 0, "syntactic_head_ID": 4, "dependency_relation": "nsubj", "word": "Darcy"},
        {"token_ID_within_document": 6, "sentence_ID": 0, "syntactic_head_ID": 4, "dependency_relation": "prep", "word": "to"},
        {"token_ID_within_document": 7, "sentence_ID": 0, "syntactic_head_ID": 6, "dependency_relation": "pobj", "word": "Elizabeth"},
    ]
    
    quote = {
        "quote_start": 0,
        "quote_end": 2,
        "mention_start": 5,
        "mention_end": 5,
        "char_id": 100
    }
    
    extractor = AddresseeExtractor(tokens, base_entities)
    interaction = extractor.extract(quote, quote_id=1)
    
    assert interaction.speaker_id == 100
    assert interaction.addressee_id == 200
    assert interaction.extraction_method == ExtractionMethod.SPEECH_TAG_OBJECT

def test_extract_vocative_external():
    base_entities = get_base_entities()
    # "Elizabeth," said Darcy
    tokens = [
        {"token_ID_within_document": 4, "sentence_ID": 0, "syntactic_head_ID": 0, "dependency_relation": "ROOT", "word": "said"},
        {"token_ID_within_document": 5, "sentence_ID": 0, "syntactic_head_ID": 4, "dependency_relation": "nsubj", "word": "Darcy"},
        {"token_ID_within_document": 7, "sentence_ID": 0, "syntactic_head_ID": 4, "dependency_relation": "npadvmod", "word": "Elizabeth"},
    ]
    
    quote = {
        "quote_start": 0,
        "quote_end": 2,
        "mention_start": 5,
        "mention_end": 5,
        "char_id": 100
    }
    
    extractor = AddresseeExtractor(tokens, base_entities)
    interaction = extractor.extract(quote, quote_id=2)
    
    assert interaction.speaker_id == 100
    assert interaction.addressee_id == 200
    assert interaction.extraction_method == ExtractionMethod.VOCATIVE

def test_extract_vocative_internal():
    base_entities = get_base_entities()
    # "Elizabeth, come here," said Darcy
    tokens = [
        {"token_ID_within_document": 1, "sentence_ID": 0, "syntactic_head_ID": 1, "dependency_relation": "ROOT", "word": "come"},
        {"token_ID_within_document": 2, "sentence_ID": 0, "syntactic_head_ID": 1, "dependency_relation": "npadvmod", "word": "Elizabeth"},
    ]
    
    quote = {
        "quote_start": 0,
        "quote_end": 3,
        "mention_start": -1,
        "mention_end": -1,
        "char_id": 100
    }
    
    extractor = AddresseeExtractor(tokens, base_entities)
    interaction = extractor.extract(quote, quote_id=3)
    
    assert interaction.speaker_id == 100
    assert interaction.addressee_id == 200
    assert interaction.extraction_method == ExtractionMethod.VOCATIVE

def test_extract_unknown():
    # "Hello," said Darcy
    tokens = [
        {"token_ID_within_document": 4, "sentence_ID": 0, "syntactic_head_ID": 0, "dependency_relation": "ROOT", "word": "said"},
        {"token_ID_within_document": 5, "sentence_ID": 0, "syntactic_head_ID": 4, "dependency_relation": "nsubj", "word": "Darcy"},
    ]
    entities = [{"COREF": "100", "start_token": "5", "end_token": "5"}]
    quote = {
        "quote_start": 0,
        "quote_end": 2,
        "mention_start": 5,
        "mention_end": 5,
        "char_id": 100
    }
    
    extractor = AddresseeExtractor(tokens, entities)
    interaction = extractor.extract(quote, quote_id=4)
    
    assert interaction.speaker_id == 100
    assert interaction.addressee_id is None
    assert interaction.extraction_method == ExtractionMethod.UNKNOWN
    assert interaction.confidence == 0.0

def test_missing_entities():
    tokens = [
        {"token_ID_within_document": 4, "sentence_ID": 0, "syntactic_head_ID": 0, "dependency_relation": "ROOT", "word": "said"},
        {"token_ID_within_document": 5, "sentence_ID": 0, "syntactic_head_ID": 4, "dependency_relation": "nsubj", "word": "Darcy"},
        {"token_ID_within_document": 6, "sentence_ID": 0, "syntactic_head_ID": 4, "dependency_relation": "prep", "word": "to"},
        {"token_ID_within_document": 7, "sentence_ID": 0, "syntactic_head_ID": 6, "dependency_relation": "pobj", "word": "Elizabeth"},
    ]
    
    # Missing Elizabeth from entities
    entities = [{"COREF": "100", "start_token": "5", "end_token": "5"}]
    quote = {
        "quote_start": 0,
        "quote_end": 2,
        "mention_start": 5,
        "mention_end": 5,
        "char_id": 100
    }
    
    extractor = AddresseeExtractor(tokens, entities)
    interaction = extractor.extract(quote, quote_id=5)
    
    # Even though we saw "to Elizabeth", we couldn't map it to a char_id
    assert interaction.addressee_id is None
    assert interaction.extraction_method == ExtractionMethod.UNKNOWN

if __name__ == "__main__":
    test_extract_speech_tag_object()
    test_extract_vocative_external()
    test_extract_vocative_internal()
    test_extract_unknown()
    test_missing_entities()
    print("All extraction tests passed!")
