import pandas as pd
from typing import Dict, List
import os
import math

from .schemas import Mention, CanonicalEntity, Quote

class BookNLPParser:
    """Parses BookNLP output files into internal semantic representations."""

    def parse_entities(self, filepath: str) -> Dict[int, CanonicalEntity]:
        """
        Parses a BookNLP .entities file into a mapping of chain_id -> CanonicalEntity.
        Raises ValueError if the file is malformed or violates constraints.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Entities file not found: {filepath}")

        try:
            df = pd.read_csv(filepath, sep='\t')
        except Exception as e:
            raise ValueError(f"Malformed entity file: {e}")

        # Check required columns
        required_cols = {'COREF', 'start_token', 'end_token', 'prop', 'cat', 'text'}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"Missing required columns in entities file. Found: {df.columns}")

        entities_by_chain: Dict[int, CanonicalEntity] = {}
        all_mention_spans = []

        for idx, row in df.iterrows():
            # Handle missing entity IDs (e.g., NaN coref)
            if pd.isna(row['COREF']) or math.isnan(row['COREF']):
                raise ValueError(f"Missing entity ID (COREF) at row {idx}")
            
            chain_id = int(row['COREF'])
            start_token = int(row['start_token'])
            end_token = int(row['end_token'])

            # Validate offsets
            if start_token > end_token:
                raise ValueError(f"Invalid offsets: start_token ({start_token}) > end_token ({end_token})")
            if start_token < 0 or end_token < 0:
                raise ValueError(f"Invalid offsets: offsets cannot be negative.")

            # BookNLP frequently outputs nested/overlapping mentions (e.g., 'his sister' vs 'his').
            # We allow overlapping mentions in the internal representation.
            all_mention_spans.append((start_token, end_token))

            mention = Mention(
                start_token=start_token,
                end_token=end_token,
                text=str(row['text']),
                prop=str(row['prop']),
                cat=str(row['cat'])
            )

            if chain_id not in entities_by_chain:
                entities_by_chain[chain_id] = CanonicalEntity(chain_id=chain_id, mentions=[])
            
            entities_by_chain[chain_id].mentions.append(mention)

        # Check for empty chains (though impossible from the loop logic unless there's a pre-declared empty chain, 
        # but we can enforce it as a sanity check)
        for chain_id, entity in entities_by_chain.items():
            if len(entity.mentions) == 0:
                raise ValueError(f"Empty chain detected for ID {chain_id}")

        return entities_by_chain

    def parse_quotes(self, filepath: str) -> List[Quote]:
        """
        Parses a BookNLP .quotes file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Quotes file not found: {filepath}")

        try:
            df = pd.read_csv(filepath, sep='\t')
        except Exception as e:
            raise ValueError(f"Malformed quote file: {e}")

        # Ensure required columns
        if 'quote_start' not in df.columns or 'quote_end' not in df.columns or 'quote' not in df.columns:
            raise ValueError("Missing quote alignment columns.")

        quotes = []
        for idx, row in df.iterrows():
            if pd.isna(row['quote_start']) or pd.isna(row['quote_end']):
                raise ValueError("Missing quote alignment (start/end token is NaN).")

            start_token = int(row['quote_start'])
            end_token = int(row['quote_end'])
            
            if start_token > end_token:
                raise ValueError(f"Invalid offsets for quote: {start_token} > {end_token}")

            char_id = None
            if 'char_id' in df.columns and not pd.isna(row['char_id']):
                char_id = int(row['char_id'])
                
            quote = Quote(
                start_token=start_token,
                end_token=end_token,
                text=str(row['quote']),
                speaker_chain_id=char_id
            )
            quotes.append(quote)
            
        return quotes

    def parse_book_aliases(self, filepath: str) -> Dict[int, set]:
        """
        Parses a .book file and returns a mapping from chain_id to a set of proper noun aliases.
        """
        import json
        aliases = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for char in data.get('characters', []):
                char_id = char.get('id')
                if char_id is None:
                    continue
                    
                alias_set = set()
                mentions = char.get('mentions', {})
                for proper in mentions.get('proper', []):
                    name = proper.get('n')
                    if name:
                        alias_set.add(name)
                        
                aliases[char_id] = alias_set
        except (FileNotFoundError, json.JSONDecodeError):
            pass
            
        return aliases
