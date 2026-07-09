from typing import List, Dict, Any, Optional

class ConversationStateModule:
    """
    Maintains the state of active conversations across quotes, tracking
    interruption distances and a participant stack for multi-party tracking.
    """
    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        self.active_conversation_id = 0
        self.participant_stack: List[str] = []
        self.last_quote_end_byte = -1
        
    def update(self, quote: Dict[str, Any], predicted_speaker: Optional[str]):
        """
        Updates the internal state based on the current quote and its predicted speaker.
        """
        # Update Participant Stack
        if predicted_speaker and predicted_speaker != "Unknown":
            # If they are already in the stack, move them to the top
            if predicted_speaker in self.participant_stack:
                self.participant_stack.remove(predicted_speaker)
            self.participant_stack.append(predicted_speaker)
            
            # Keep stack size bounded (e.g., last 5 active participants)
            if len(self.participant_stack) > 5:
                self.participant_stack.pop(0)
                
        # Update the end byte for the next quote to measure distance
        end_byte = quote.get("quote_end_byte", -1)
        if end_byte != -1:
            self.last_quote_end_byte = end_byte
            
    def get_interruption_distance(self, current_quote_start_byte: int) -> int:
        if self.last_quote_end_byte == -1 or current_quote_start_byte == -1:
            return 0
        distance = current_quote_start_byte - self.last_quote_end_byte
        return distance if distance > 0 else 0

    def reset(self, scene_id: str):
        """
        Hard reset for a completely new scene.
        """
        self.scene_id = scene_id
        self.active_conversation_id += 1
        self.participant_stack = []
        self.last_quote_end_byte = -1
