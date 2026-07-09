from typing import List, Dict, Optional, Any
from .schemas import DialogueInteraction, ExtractionMethod

class AddresseeExtractor:
    def __init__(self, tokens: List[Dict], entities: List[Dict], alias_dict: Dict = None):
        self.doc_tokens = {int(t['token_ID_within_document']): t for t in tokens}
        self.alias_dict = alias_dict or {}
        
        self.token_to_char = {}
        for ent in entities:
            start = int(ent.get('start_token', -1))
            end = int(ent.get('end_token', -1))
            char_id = ent.get('COREF') or ent.get('char_id')
            if char_id and start != -1 and end != -1:
                try:
                    char_id = int(char_id)
                    for t_id in range(start, end + 1):
                        self.token_to_char[t_id] = char_id
                except ValueError:
                    pass

    def extract(self, quote: Dict, quote_id: int) -> DialogueInteraction:
        q_start = int(quote.get('quote_start', -1))
        q_end = int(quote.get('quote_end', -1))
        m_start = int(quote.get('mention_start', -1))
        m_end = int(quote.get('mention_end', -1))
        
        try:
            speaker_id = int(quote.get('char_id', -1))
        except ValueError:
            speaker_id = -1
            
        candidates = [] # list of (char_id, method, confidence)
        
        # 1. Look for Speech Tag Objects and external Vocatives
        if m_start >= 0 and m_end >= 0:
            head_verb_id = -1
            
            # Find the head verb of the speaker mention
            for t_id in range(m_start, m_end + 1):
                if t_id in self.doc_tokens:
                    tok = self.doc_tokens[t_id]
                    try:
                        head_doc_id = int(tok.get('syntactic_head_ID', 0))
                    except ValueError:
                        continue
                    if head_doc_id != t_id and head_doc_id in self.doc_tokens:
                        if not (m_start <= head_doc_id <= m_end):
                            head_verb_id = head_doc_id
                            break
                            
            if head_verb_id != -1:
                verb_tok = self.doc_tokens[head_verb_id]
                sent_id = verb_tok.get('sentence_ID')
                
                for t_id, tok in self.doc_tokens.items():
                    if tok.get('sentence_ID') == sent_id:
                        try:
                            head_idx = int(tok.get('syntactic_head_ID', 0))
                        except ValueError:
                            continue
                            
                        # If attached to the speech verb
                        if head_idx == head_verb_id:
                            # Rule 1: Speech Tag Object
                            if tok.get('dependency_relation') == 'prep' and tok.get('word', '').lower() in ['to', 'with', 'at']:
                                for t2_id, tok2 in self.doc_tokens.items():
                                    if tok2.get('sentence_ID') == sent_id:
                                        try:
                                            child_head_idx = int(tok2.get('syntactic_head_ID', 0))
                                        except ValueError:
                                            continue
                                        if child_head_idx == t_id and tok2.get('dependency_relation') == 'pobj':
                                            char_id = self.token_to_char.get(t2_id)
                                            if char_id:
                                                candidates.append((char_id, ExtractionMethod.SPEECH_TAG_OBJECT, 0.9))
                            
                            # Rule 2: External Vocative
                            if tok.get('dependency_relation') == 'npadvmod':
                                char_id = self.token_to_char.get(t_id)
                                if char_id:
                                    candidates.append((char_id, ExtractionMethod.VOCATIVE, 0.6))

        # 2. Look for Internal Vocatives (inside the quote)
        if q_start >= 0 and q_end >= 0:
            for t_id in range(q_start, q_end + 1):
                if t_id in self.doc_tokens:
                    tok = self.doc_tokens[t_id]
                    if tok.get('dependency_relation') == 'npadvmod':
                        char_id = self.token_to_char.get(t_id)
                        if char_id:
                            candidates.append((char_id, ExtractionMethod.VOCATIVE, 0.5))

        # Pick the candidate with the highest confidence
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            best_candidate = candidates[0]
            
            # Resolve aliases if needed
            final_addressee = best_candidate[0]
            if str(final_addressee) in self.alias_dict:
                final_addressee = self.alias_dict[str(final_addressee)]
                
            return DialogueInteraction(
                quote_id=quote_id,
                speaker_id=speaker_id,
                addressee_id=final_addressee,
                confidence=best_candidate[2],
                extraction_method=best_candidate[1]
            )

        # Unknown fallback
        return DialogueInteraction(
            quote_id=quote_id,
            speaker_id=speaker_id,
            addressee_id=None,
            confidence=0.0,
            extraction_method=ExtractionMethod.UNKNOWN
        )
