from dataclasses import dataclass

@dataclass
class CharacterFingerprint:
    character_id: str
    quotes_seen: int
    total_tokens: int
    texts: list[str]  # Experimental representation only. Replace with incremental statistics if accepted.

@dataclass
class StyleState:
    fingerprints: dict[str, CharacterFingerprint]
