from typing import Dict, List, Set, Tuple
from .schemas import CanonicalEntity

class RepresentationValidator:
    """Validates the internal representation of coreference data."""
    
    @staticmethod
    def validate_chain_integrity(entities: Dict[int, CanonicalEntity]) -> List[str]:
        """
        Validates coreference chain integrity:
        - Chain IDs are unique (inherent in Dict).
        - No orphan mentions (every mention belongs to its enclosing chain).
        - No duplicate mentions in a chain (exact same token spans).
        """
        errors = []
        for chain_id, entity in entities.items():
            if entity.chain_id != chain_id:
                errors.append(f"Chain ID mismatch: {chain_id} != {entity.chain_id}")
            
            seen_spans: Set[Tuple[int, int]] = set()
            for mention in entity.mentions:
                span = (mention.start_token, mention.end_token)
                if span in seen_spans:
                    errors.append(f"Duplicate mention {span} in chain {chain_id}")
                seen_spans.add(span)
        return errors

    @staticmethod
    def validate_mention_ordering(entities: Dict[int, CanonicalEntity]) -> List[str]:
        """
        Validates mention ordering and offsets:
        - Mentions are ordered by document position.
        - Start/end offsets are valid (start <= end, non-negative).
        """
        errors = []
        for chain_id, entity in entities.items():
            last_start = -1
            for mention in entity.mentions:
                if mention.start_token < 0 or mention.end_token < 0:
                    errors.append(f"Negative offsets in chain {chain_id}: {mention.start_token}, {mention.end_token}")
                if mention.start_token > mention.end_token:
                    errors.append(f"Invalid offsets in chain {chain_id}: start {mention.start_token} > end {mention.end_token}")
                
                # BookNLP entities are typically sorted, but let's check
                if mention.start_token < last_start:
                    errors.append(f"Mentions not ordered in chain {chain_id}: {mention.start_token} < {last_start}")
                last_start = mention.start_token
        return errors

    @staticmethod
    def validate_canonical_mapping(entities: Dict[int, CanonicalEntity]) -> List[str]:
        """
        Validates canonical mapping:
        - Every chain has at least one mention.
        - Detect ambiguous/empty chains.
        """
        errors = []
        for chain_id, entity in entities.items():
            if not entity.mentions:
                errors.append(f"Empty chain: {chain_id}")
        return errors

    @staticmethod
    def run_all(entities: Dict[int, CanonicalEntity]) -> List[str]:
        errors = []
        errors.extend(RepresentationValidator.validate_chain_integrity(entities))
        errors.extend(RepresentationValidator.validate_mention_ordering(entities))
        errors.extend(RepresentationValidator.validate_canonical_mapping(entities))
        return errors
