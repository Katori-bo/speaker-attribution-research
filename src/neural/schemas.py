from dataclasses import dataclass
import torch
from typing import List

@dataclass
class CandidateExample:
    candidate_id: str
    features: torch.Tensor
    is_gold: bool

@dataclass
class QuoteStep:
    quote_id: str
    candidates: List[CandidateExample]
    gold_speaker: str

@dataclass
class NovelSequence:
    novel_id: str
    quotes: List[QuoteStep]
