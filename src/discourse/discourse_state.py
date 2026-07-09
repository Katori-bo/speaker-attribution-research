from typing import List, Set, Optional, Dict

class MinimalDiscourseState:
    """
    Maintains the minimal state required for the symbolic baseline attribution rules.
    No probabilities, embeddings, or ML components.
    """
    def __init__(self):
        self.last_speaker: Optional[str] = None
        self.previous_speaker: Optional[str] = None
        self.recent_mentions: List[str] = []
        self.current_candidates: Set[str] = set()
        self.dialogue_position: int = 0
        self.conversation_length: int = 0
        # EXP017B: Soft probability distribution over candidates for last speaker
        self.last_speaker_probs: Dict[str, float] = {}
        
    def update(self, speaker: str, mentions: List[str], candidates: Set[str]):
        """
        Updates the discourse state after a quote is attributed.
        """
        if speaker:
            self.previous_speaker = self.last_speaker
            self.last_speaker = speaker
            
        self.recent_mentions = mentions[-5:] # Keep only the most recent 5 mentions
        self.current_candidates = candidates
        
        self.dialogue_position += 1
        self.conversation_length += 1
        
    def reset_conversation(self):
        """
        Resets conversational tracking (e.g., at a scene break or chapter end).
        """
        self.last_speaker = None
        self.previous_speaker = None
        self.dialogue_position = 0
        self.conversation_length = 0
        self.last_speaker_probs = {}
