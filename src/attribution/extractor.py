import re
from typing import Optional
import spacy

from src.attribution.schemas import SpeechAttribution, ExtractionMethod

class AttributionExtractor:
    """
    Extracts explicit speaker attributions from quote contexts using 
    quote-aligned dependency parsing. Strictly limits extraction to 
    NAMED speakers (PROPN).
    """
    def __init__(self):
        # We assume spacy model is downloaded
        self.nlp = spacy.load("en_core_web_sm")
        self.speech_verbs = {
            "say", "ask", "reply", "cry", "answer", "exclaim", 
            "whisper", "shout", "add", "continue", "observe", 
            "remark", "murmur", "think"
        }
        
    def extract(self, quote_id: str, quote_start: int, quote_end: int, content: str) -> Optional[SpeechAttribution]:
        left_context = content[max(0, quote_start - 150):quote_start]
        right_context = content[quote_end:min(len(content), quote_end + 150)]
        
        # Region 1: Right attached tag
        match_right = re.search(r'[\.\!\?\n]', right_context)
        if match_right:
            right_span = right_context[:match_right.end()]
        else:
            right_span = right_context
            
        # Region 2: Left attached tag
        match_left = re.search(r'[\.\!\?\n]', left_context[::-1])
        if match_left:
            left_span = left_context[-match_left.end():]
        else:
            left_span = left_context
            
        right_span = right_span.strip()
        left_span = left_span.strip()
        
        best_verb = None
        best_nsubj = None
        
        # Parse Region 1
        if right_span:
            doc_right = self.nlp(right_span)
            for token in doc_right:
                if token.lemma_.lower() in self.speech_verbs:
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            best_nsubj = child
                            best_verb = token
                            break
                    if best_nsubj:
                        break
                        
        # Parse Region 2 if Region 1 failed
        if not best_nsubj and left_span:
            doc_left = self.nlp(left_span)
            for token in reversed(doc_left):
                if token.lemma_.lower() in self.speech_verbs:
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            best_nsubj = child
                            best_verb = token
                            break
                    if best_nsubj:
                        break
                        
        if best_nsubj is not None:
            speaker_tokens = [best_nsubj]
            for child in best_nsubj.children:
                if child.dep_ in ("compound", "det", "amod"):
                    speaker_tokens.append(child)
            speaker_tokens = sorted(speaker_tokens, key=lambda x: x.i)
            
            # STRICT REQUIREMENT: Must contain a PROPN (no uppercase fallback)
            is_named = any(token.pos_ == "PROPN" for token in speaker_tokens)
            if not is_named:
                return None
                
            speaker_mention = " ".join([t.text for t in speaker_tokens])
            
            return SpeechAttribution(
                quote_id=quote_id,
                speaker_mention=speaker_mention,
                extraction_method=ExtractionMethod.NAMED_SYNTACTIC,
                confidence=1.0  # Confirmed fixed precision
            )
            
        return None
