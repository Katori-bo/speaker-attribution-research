from enum import Enum
from typing import Optional
import pandas as pd

class QuoteType(Enum):
    EXPLICIT_NAMED = "Explicit named"
    EXPLICIT_NOMINAL = "Explicit nominal"
    EXPLICIT_PRONOUN = "Explicit pronoun"
    IMPLICIT = "Implicit"

class QuoteClassifier:
    """
    Classifies quotes into explicit/implicit types based on BookNLP attribution.
    """
    def __init__(self):
        self.pronouns = {"he", "she", "they", "i", "we", "you", "him", "her", "them", "us", "me"}

    def classify(self, mention_phrase: Optional[str], quote_start: int, quote_end: int, mention_start: int, mention_end: int) -> QuoteType:
        if not mention_phrase or mention_start is None or mention_start == -1 or pd.isna(mention_start):
            return QuoteType.IMPLICIT
            
        m_phrase = str(mention_phrase).strip()
        if m_phrase == "" or m_phrase.lower() == "nan":
            return QuoteType.IMPLICIT
            
        # Distance heuristic: if mention is far from the quote, BookNLP probably
        # resolved an implicit conversational chain using a previous mention.
        dist = min(abs(mention_start - quote_end), abs(quote_start - mention_end))
        if dist > 15:
            return QuoteType.IMPLICIT
            
        if m_phrase.lower() in self.pronouns:
            return QuoteType.EXPLICIT_PRONOUN
            
        if any(c.isupper() for c in m_phrase):
            return QuoteType.EXPLICIT_NAMED
            
        return QuoteType.EXPLICIT_NOMINAL
