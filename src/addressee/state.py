from typing import List, Dict, Optional
from .schemas import InteractionState, DialogueInteraction

class InteractionStateUpdater:
    def __init__(self, history_limit: int = 5):
        self.state = InteractionState()
        self.history_limit = history_limit

    def update(self, interaction: DialogueInteraction) -> InteractionState:
        # Update transition history if there was a previous interaction
        if self.state.last_interaction:
            prev_speaker = self.state.last_interaction.speaker_id
            curr_speaker = interaction.speaker_id
            
            # Record the transition if both speakers are known
            if prev_speaker != -1 and curr_speaker != -1:
                self.state.speaker_transition_history.append({
                    "from": prev_speaker,
                    "to": curr_speaker
                })
                
                # Keep history within limit
                if len(self.state.speaker_transition_history) > self.history_limit:
                    self.state.speaker_transition_history = self.state.speaker_transition_history[-self.history_limit:]
        
        # Update recent addressees
        if interaction.addressee_id is not None:
            # Bump to front if already exists, else add
            if interaction.addressee_id in self.state.recent_addressees:
                self.state.recent_addressees.remove(interaction.addressee_id)
            self.state.recent_addressees.insert(0, interaction.addressee_id)
            
            # Keep recent addressees within limit
            if len(self.state.recent_addressees) > self.history_limit:
                self.state.recent_addressees = self.state.recent_addressees[:self.history_limit]
                
            # Track when they were last addressed
            self.state.last_addressed_at[interaction.addressee_id] = interaction.quote_id
                
        # Update last interaction
        self.state.last_interaction = interaction
        
        return self.state
