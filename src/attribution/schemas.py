from enum import Enum
from dataclasses import dataclass

class ExtractionMethod(Enum):
    NAMED_SYNTACTIC = "NAMED_SYNTACTIC"
    UNKNOWN = "UNKNOWN"

@dataclass
class SpeechAttribution:
    quote_id: str
    speaker_mention: str
    extraction_method: ExtractionMethod
    confidence: float
