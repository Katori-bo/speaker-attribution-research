from typing import List, Callable, Tuple, Optional, Dict, Any
from src.discourse.discourse_state import MinimalDiscourseState

class RuleEvaluator:
    """
    Independently measures ALL rules against a quote to determine which fire 
    and whether they recommend a speaker.
    """
    def __init__(self, rules: List[Tuple[str, Callable]]):
        self.rules = rules
        
    def evaluate(self, quote_text: str, context_text: str, state: MinimalDiscourseState, explicit_mentions: list) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Returns a dictionary mapping Rule Name -> (Applicable, Predicted Speaker).
        """
        results = {}
        for rule_name, rule_func in self.rules:
            applicable, prediction = rule_func(quote_text, context_text, state, explicit_mentions)
            results[rule_name] = (applicable, prediction)
        return results

class RuleEngine:
    """
    Receives evaluations from the RuleEvaluator and emits a final prediction 
    based strictly on priority ordering.
    """
    def __init__(self, rules: List[Tuple[str, Callable]]):
        self.priority_order = [r[0] for r in rules]
        
    def decide(self, evaluations: Dict[str, Tuple[bool, Optional[str]]]) -> Tuple[str, str]:
        """
        Returns (Predicted_Speaker, Winning_Rule_Name).
        Falls back to 'Unknown' if no rule fired.
        """
        for rule_name in self.priority_order:
            applicable, pred = evaluations.get(rule_name, (False, None))
            if applicable and pred:
                return pred, rule_name
                
        return "Unknown", "No Match"
