from typing import Dict, Any, List
from src.discourse.discourse_state import MinimalDiscourseState
from src.baseline.candidate_generator import CandidateGenerator
from src.baseline.rule_engine import RuleEngine
from src.baseline.attribution_rules import ATTRIBUTION_RULES

class SymbolicAttributor:
    """
    Orchestrates the Candidate Generator, Discourse State, and Rule Engine 
    to perform speaker attribution for a sequence of quotes.
    """
    def __init__(self):
        self.state = MinimalDiscourseState()
        self.candidate_generator = CandidateGenerator()
        self.rule_engine = RuleEngine(ATTRIBUTION_RULES)
        
    def attribute_sequence(self, quotes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a sequence of quotes sequentially, updating state at each step.
        """
        self.state.reset_conversation()
        results = []
        
        for quote in quotes:
            text = quote.get("text", "")
            context = quote.get("context", "")
            
            # Simple mock parsing for candidates based on provided inputs
            explicit = quote.get("explicit_mentions", [])
            nearby = quote.get("nearby_characters", [])
            paragraph = quote.get("paragraph_mentions", [])
            
            candidates = self.candidate_generator.generate_candidates(
                explicit_mentions=explicit,
                nearby_characters=nearby,
                previous_participants=[self.state.last_speaker, self.state.previous_speaker] if self.state.last_speaker else [],
                local_paragraph_mentions=paragraph
            )
            
            self.state.current_candidates = candidates
            
            prediction_result = self.rule_engine.predict(text, context, self.state)
            
            # Update state with prediction and recent mentions
            self.state.update(
                speaker=prediction_result["prediction"],
                mentions=explicit + nearby + paragraph,
                candidates=candidates
            )
            
            results.append({
                "quote_id": quote.get("quote_id", "unknown"),
                "prediction": prediction_result["prediction"],
                "reason": prediction_result["reason"]
            })
            
        return results
