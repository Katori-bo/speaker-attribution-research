from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum

class ExtractionMethod(Enum):
    VOCATIVE = "VOCATIVE"
    SPEECH_TAG_OBJECT = "SPEECH_TAG_OBJECT"
    UNKNOWN = "UNKNOWN"

@dataclass
class DialogueInteraction:
    quote_id: int
    speaker_id: int
    addressee_id: Optional[int]
    confidence: float
    extraction_method: ExtractionMethod

@dataclass
class InteractionState:
    last_interaction: Optional[DialogueInteraction] = None
    recent_addressees: List[int] = field(default_factory=list)
    speaker_transition_history: List[Dict[str, int]] = field(default_factory=list)
    last_addressed_at: Dict[int, int] = field(default_factory=dict)
