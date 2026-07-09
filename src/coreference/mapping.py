from typing import Dict, Optional, Tuple
from .schemas import CanonicalEntity

class MentionToEntityMapper:
    """
    Provides mapping resolution from mentions to their canonical entity chains.
    """
    def __init__(self, entities: Dict[int, CanonicalEntity], aliases: Dict[int, set] = None):
        self.entities = entities
        self.aliases = aliases if aliases is not None else {}
        self._build_mention_index()

    def _build_mention_index(self):
        self.mention_index: Dict[Tuple[int, int], CanonicalEntity] = {}
        for chain_id, entity in self.entities.items():
            for mention in entity.mentions:
                span = (mention.start_token, mention.end_token)
                # Overlaps are handled at the parser level; duplicate exact spans
                # are also not expected, but if they exist, the last one overwrites.
                self.mention_index[span] = entity
    
    def resolve_mention(self, start_token: int, end_token: int) -> Optional[CanonicalEntity]:
        """
        Resolves an exact mention span to its canonical entity.
        Returns None if no entity is found for the given span.
        """
        return self.mention_index.get((start_token, end_token))
        
    def get_entity(self, chain_id: int) -> Optional[CanonicalEntity]:
        """Returns the canonical entity given its chain ID."""
        return self.entities.get(chain_id)

    def resolve_string_to_chain_ids(self, candidate_str: str) -> list[int]:
        """
        Returns a list of all chain_ids that match the candidate string, 
        to help identify ambiguous mappings.
        """
        candidate_str_lower = candidate_str.lower().strip()
        matched_chains = set()
        
        # 1. Exact match against aliases
        for chain_id, alias_set in self.aliases.items():
            for alias in alias_set:
                if alias.lower().strip() == candidate_str_lower:
                    matched_chains.add(chain_id)
                    
        if matched_chains:
            return list(matched_chains)
            
        # 2. Subset match
        max_overlap = 0
        best_chains = set()
        
        for chain_id, alias_set in self.aliases.items():
            for alias in alias_set:
                alias_lower = alias.lower().strip()
                if not alias_lower or len(alias_lower) < 3:
                    continue
                    
                if alias_lower in candidate_str_lower or candidate_str_lower in alias_lower:
                    overlap_len = min(len(alias_lower), len(candidate_str_lower))
                    if overlap_len > max_overlap:
                        max_overlap = overlap_len
                        best_chains = {chain_id}
                    elif overlap_len == max_overlap and max_overlap > 0:
                        best_chains.add(chain_id)
                        
        return list(best_chains)

    def resolve_string_to_chain_id(self, candidate_str: str) -> Optional[int]:
        """
        Resolves a string to a single chain_id (the first one found or the best match).
        """
        chains = self.resolve_string_to_chain_ids(candidate_str)
        return chains[0] if chains else None
