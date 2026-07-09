import os
import csv
import sys
import pandas as pd
from typing import Dict, Any

from src.addressee.extractor import AddresseeExtractor
from src.addressee.state import InteractionStateUpdater
from src.addressee.features import extract_addressee_features
from src.coreference.parser import BookNLPParser
from src.coreference.mapping import MentionToEntityMapper

# Increase csv limits
csv.field_size_limit(sys.maxsize)

class AddresseeFeatureProvider:
    """
    Integrates the addressee feature extractors into the ranking pipeline.
    Maintains dialogue state sequentially per quote.
    """
    def __init__(self, booknlp_out_dir: str = "data/raw/pdnc/booknlp_out", enabled: bool = True):
        self.booknlp_out_dir = booknlp_out_dir
        self.enabled = enabled
        
        self.current_novel = None
        self.extractor = None
        self.updater = None
        self.mapper = None
        self.quotes_list = []
        
    def _load_novel(self, novel: str):
        if self.current_novel == novel:
            return
            
        novel_dir = os.path.join(self.booknlp_out_dir, novel)
        entities_path = os.path.join(novel_dir, f"{novel}.entities")
        tokens_path = os.path.join(novel_dir, f"{novel}.tokens")
        quotes_path = os.path.join(novel_dir, f"{novel}.quotes")
        book_path = os.path.join(novel_dir, f"{novel}.book")
        
        if not all(os.path.exists(p) for p in [entities_path, tokens_path, quotes_path, book_path]):
            raise FileNotFoundError(f"BookNLP outputs missing for novel {novel}")
            
        parser = BookNLPParser()
        entities = parser.parse_entities(entities_path)
        aliases = parser.parse_book_aliases(book_path)
        self.mapper = MentionToEntityMapper(entities, aliases)
        
        tokens = []
        with open(tokens_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                tokens.append(row)
                
        raw_entities = []
        with open(entities_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                raw_entities.append(row)
                
        self.quotes_list = []
        with open(quotes_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                self.quotes_list.append(row)
                
        self.extractor = AddresseeExtractor(tokens, raw_entities, alias_dict={})
        self.updater = InteractionStateUpdater(history_limit=10)
        self.current_novel = novel

    def reset_state(self):
        if self.updater:
            self.updater = InteractionStateUpdater(history_limit=10)

    def extract(self, novel: str, quote_index: int, candidate_str: str) -> Dict[str, Any]:
        """
        Extracts features for a given candidate at the current state.
        This does NOT update the state.
        """
        if not self.enabled:
            return {
                "candidate_was_addressed": 0.0,
                "addressee_recency": -1.0,
                "speaker_addressee_transition": 0.0
            }
            
        self._load_novel(novel)
        
        chain_id = self.mapper.resolve_string_to_chain_id(candidate_str)
        if chain_id is None:
            return {
                "candidate_was_addressed": 0.0,
                "addressee_recency": -1.0,
                "speaker_addressee_transition": 0.0
            }
            
        features = extract_addressee_features(int(chain_id), quote_index, self.updater.state)
        
        return {
            "candidate_was_addressed": 1.0 if features["candidate_was_addressed"] else 0.0,
            "addressee_recency": float(features["addressee_recency"]),
            "speaker_addressee_transition": 1.0 if features["speaker_addressee_transition"] else 0.0
        }
        
    def update_state(self, novel: str, quote_index: int, true_speaker_str: str = None):
        """
        Updates the addressee interaction state based on the current quote.
        Normally, true_speaker is what the model predicted, or gold during training.
        Wait, our extractor extracts the speaker from the BookNLP quote metadata directly.
        So we just process the quote at quote_index.
        """
        if not self.enabled:
            return
            
        self._load_novel(novel)
        
        # In PDNC, we often iterate sequentially. 
        # For simplicity, we just use the BookNLP quote at this index.
        # Ensure quote_index is within bounds (though BookNLP vs PDNC quote alignment is roughly 1:1, there can be slight mismatches.
        # But we assume quote_index is valid BookNLP quote index.)
        if 0 <= quote_index < len(self.quotes_list):
            quote = self.quotes_list[quote_index]
            interaction = self.extractor.extract(quote, quote_index)
            
            # If the extraction failed to find speaker, maybe use true_speaker_str if provided?
            # But the baseline EXP013 extracts everything from BookNLP automatically.
            
            self.updater.update(interaction)
