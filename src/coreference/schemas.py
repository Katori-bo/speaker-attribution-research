from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Mention:
    """A single reference to an entity in the text, using BookNLP token offsets."""
    start_token: int
    end_token: int
    text: str
    prop: str
    cat: str

@dataclass
class CanonicalEntity:
    """A unique character or entity, aggregating all its coreferential mentions."""
    chain_id: int
    mentions: List[Mention]

@dataclass
class Quote:
    """A quotation identified by BookNLP, using token offsets."""
    start_token: int
    end_token: int
    text: str
    speaker_chain_id: Optional[int] = None
