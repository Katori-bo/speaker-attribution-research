from typing import List, Dict, Tuple, Any
from .schemas import DialogueInteraction, ExtractionMethod

class ValidationEngine:
    def __init__(self, gold_annotations: Dict[int, List[str]], alias_dict: Dict[str, int]):
        """
        gold_annotations: dict mapping quote_id to a list of addressee names or IDs
        alias_dict: maps string names to char_ids
        """
        self.gold_annotations = gold_annotations
        self.alias_dict = alias_dict
        
    def validate(self, interactions: List[DialogueInteraction]) -> Dict[str, Any]:
        total_quotes = len(interactions)
        extracted = [i for i in interactions if i.extraction_method != ExtractionMethod.UNKNOWN]
        
        vocative_count = sum(1 for i in extracted if i.extraction_method == ExtractionMethod.VOCATIVE)
        speech_tag_count = sum(1 for i in extracted if i.extraction_method == ExtractionMethod.SPEECH_TAG_OBJECT)
        unknown_count = total_quotes - len(extracted)
        
        correct = 0
        evaluated = 0
        
        for interaction in extracted:
            # Check against gold
            gold = self.gold_annotations.get(interaction.quote_id)
            if gold:
                evaluated += 1
                
                # Simple check if the extracted addressee matches one of the gold IDs/Names
                # Since gold annotations might be character strings, we can map interaction.addressee_id to string,
                # or map gold string to char_id.
                # Assuming gold_annotations contains mapped char_ids for simplicity in this evaluation layer.
                if interaction.addressee_id in gold:
                    correct += 1
                    
        precision = correct / evaluated if evaluated > 0 else 0.0
        coverage = (len(extracted) / total_quotes) * 100 if total_quotes > 0 else 0.0
        
        return {
            "total_quotes": total_quotes,
            "extracted_addressees": len(extracted),
            "coverage_percent": coverage,
            "method_distribution": {
                "VOCATIVE": vocative_count,
                "SPEECH_TAG_OBJECT": speech_tag_count,
                "UNKNOWN": unknown_count
            },
            "confidence_distribution": {
                "high (>=0.8)": sum(1 for i in extracted if i.confidence >= 0.8),
                "medium (0.5-0.79)": sum(1 for i in extracted if 0.5 <= i.confidence < 0.8),
                "low (<0.5)": sum(1 for i in extracted if i.confidence < 0.5)
            },
            "precision_estimate": precision,
            "evaluated_against_gold": evaluated
        }
