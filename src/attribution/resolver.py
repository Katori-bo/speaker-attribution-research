from typing import Optional
from src.coreference.mapping import MentionToEntityMapper

class AttributionResolver:
    """
    Resolves extracted speaker mentions to canonical entity chain IDs.
    Strictly reuses the existing EXP012 MentionToEntityMapper.
    """
    def __init__(self, mapper: MentionToEntityMapper):
        self.mapper = mapper
        
    def resolve(self, speaker_mention: str) -> Optional[int]:
        """
        Takes an extracted string like "Darcy" and returns the best chain ID 
        using the exact existing alias resolution logic. No new fuzzy logic added.
        """
        return self.mapper.resolve_string_to_chain_id(speaker_mention)
