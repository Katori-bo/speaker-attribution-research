from typing import List, Set

class CandidateGenerator:
    """
    Generates a set of candidate speakers for a given quote based on simple proximity and mentions.
    Designed for extremely high recall.
    """
    
    def generate_candidates(
        self,
        explicit_mentions: List[str],
        nearby_characters: List[str],
        previous_participants: List[str],
        local_paragraph_mentions: List[str]
    ) -> Set[str]:
        """
        Combines various sources of characters into a unique candidate set.
        
        Args:
            explicit_mentions: Characters explicitly mentioned in the quote's text or attribution span.
            nearby_characters: Characters mentioned in the immediate surrounding sentences.
            previous_participants: Characters who spoke recently in the current conversation.
            local_paragraph_mentions: All characters mentioned in the current paragraph.
            
        Returns:
            A unique set of candidate speaker strings.
        """
        candidates = set()
        
        for source in [explicit_mentions, nearby_characters, previous_participants, local_paragraph_mentions]:
            for char in source:
                if char:
                    candidates.add(char)
                    
        return candidates
