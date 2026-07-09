from typing import Optional

class DiscourseMode:
    """Base class for discourse update strategies."""
    def __init__(self):
        self.name = "base"
        
    def resolve_speaker(self, gold_speaker: str, predicted_speaker: str, confidence: float = None) -> Optional[str]:
        raise NotImplementedError
        
class TeacherForcedMode(DiscourseMode):
    def __init__(self):
        super().__init__()
        self.name = "teacher_forced"
        
    def resolve_speaker(self, gold_speaker: str, predicted_speaker: str, confidence: float = None) -> str:
        return gold_speaker

class FullyAutoregressiveMode(DiscourseMode):
    def __init__(self):
        super().__init__()
        self.name = "fully_autoregressive"
        
    def resolve_speaker(self, gold_speaker: str, predicted_speaker: str, confidence: float = None) -> str:
        return predicted_speaker

class OneStepAutoregressiveMode(DiscourseMode):
    """
    One-step autoregressive: for quote N, only state.last_speaker is set to the
    model's predicted speaker of quote N-1. All other state (previous_speaker,
    participant_stack, dialogue_position, etc.) is built from gold labels.
    
    This isolates the effect of a single corrupted feature (candidate_is_last_speaker)
    without compounding through the full state history.
    """
    def __init__(self):
        super().__init__()
        self.name = "one_step_autoregressive"
        
    def resolve_speaker(self, gold_speaker: str, predicted_speaker: str, confidence: float = None) -> str:
        return predicted_speaker

class ConfidenceGatedMode(DiscourseMode):
    """
    Fully autoregressive, but only commits a predicted speaker to state when
    the model's confidence exceeds a threshold. Below-threshold predictions
    are treated as Unknown, preventing low-confidence errors from corrupting
    the discourse state for subsequent quotes.
    """
    def __init__(self, threshold: float = 0.85):
        super().__init__()
        self.threshold = threshold
        self.name = f"confidence_gated_{threshold:.2f}"
        
    def resolve_speaker(self, gold_speaker: str, predicted_speaker: str, confidence: float = None) -> Optional[str]:
        if confidence is not None and confidence >= self.threshold:
            return predicted_speaker
        return None  # Don't commit to state — mark as Unknown

class ReverseOneStepAutoregressiveMode(DiscourseMode):
    """
    Reverse one-step autoregressive: for quote N, corrupts the entire state with
    the predicted speaker, but then restores state.last_speaker to the gold value.
    This tests if corruption is distributed across the history features.
    """
    def __init__(self):
        super().__init__()
        self.name = "reverse_one_step_autoregressive"
        
    def resolve_speaker(self, gold_speaker: str, predicted_speaker: str, confidence: float = None) -> str:
        return predicted_speaker

class ExplicitAnchorResetMode(DiscourseMode):
    """
    Fully autoregressive, but uses explicit attribution as a state anchor.
    """
    def __init__(self):
        super().__init__()
        self.name = "explicit_anchor_reset"
        
    def resolve_speaker(self, gold_speaker: str, predicted_speaker: str, confidence: float = None) -> str:
        return predicted_speaker
