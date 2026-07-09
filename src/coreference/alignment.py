import pandas as pd
from typing import List, Tuple, Dict, Optional

class AlignmentLayer:
    """Maps byte offsets (e.g., from PDNC) to BookNLP token IDs."""
    
    def __init__(self, tokens_df: pd.DataFrame):
        tokens_df = tokens_df.copy()
        tokens_df['byte_onset'] = pd.to_numeric(tokens_df['byte_onset'], errors='coerce')
        tokens_df = tokens_df.dropna(subset=['byte_onset'])
        self.tokens = tokens_df.sort_values('byte_onset').to_dict('records')
        self._build_index()

    def _build_index(self):
        # Build an interval index or simple list for fast lookup
        # Since text is linear, a sorted list is fine.
        self.byte_to_token = {}
        for row in self.tokens:
            try:
                tok_id = int(row['token_ID_within_document'])
                start = int(row['byte_onset'])
                end = int(row['byte_offset'])
            except (ValueError, TypeError):
                continue
            for b in range(start, end):
                self.byte_to_token[b] = tok_id

    def map_byte_span_to_tokens(self, byte_start: int, byte_end: int) -> List[int]:
        """
        Maps a byte span [byte_start, byte_end) to a list of token IDs.
        Includes any token that overlaps with the byte span.
        """
        token_ids = set()
        # Byte end in PDNC is typically inclusive or exclusive? 
        # We will check every byte in the range.
        for b in range(byte_start, byte_end):
            if b in self.byte_to_token:
                token_ids.add(self.byte_to_token[b])
        return sorted(list(token_ids))

    def map_quote_byte_spans_to_tokens(self, byte_spans: List[Tuple[int, int]]) -> List[int]:
        """
        Maps a list of byte spans (e.g., a discontinuous PDNC quote) to token IDs.
        """
        token_ids = set()
        for start, end in byte_spans:
            # Assume end is inclusive in PDNC, so we do end+1 for range
            # Adjust if PDNC is exclusive. BookNLP offset is exclusive for end.
            tokens = self.map_byte_span_to_tokens(start, end + 1)
            token_ids.update(tokens)
        return sorted(list(token_ids))
