import pytest
from src.baseline.candidate_generator import CandidateGenerator

def test_candidate_generator_combines_sources_uniquely():
    generator = CandidateGenerator()
    
    explicit = ["Alice", "Bob"]
    nearby = ["Bob", "Charlie"]
    previous = ["Alice", "David"]
    paragraph = ["Eve", "Alice"]
    
    candidates = generator.generate_candidates(
        explicit_mentions=explicit,
        nearby_characters=nearby,
        previous_participants=previous,
        local_paragraph_mentions=paragraph
    )
    
    expected = {"Alice", "Bob", "Charlie", "David", "Eve"}
    assert candidates == expected

def test_candidate_generator_handles_empty_inputs():
    generator = CandidateGenerator()
    
    candidates = generator.generate_candidates(
        explicit_mentions=[],
        nearby_characters=[],
        previous_participants=[],
        local_paragraph_mentions=[]
    )
    
    assert candidates == set()
    
def test_candidate_generator_ignores_empty_strings():
    generator = CandidateGenerator()
    
    candidates = generator.generate_candidates(
        explicit_mentions=["Alice", ""],
        nearby_characters=[None, "Bob"] if None else ["Bob"],  # simplified for typing
        previous_participants=[],
        local_paragraph_mentions=[]
    )
    
    assert candidates == {"Alice", "Bob"}
