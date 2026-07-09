import random
import time
import sys
import pandas as pd
from src.style.state import CharacterStyleState
from src.style.features import compute_similarity_scores

class StyleFeatureProvider:
    def __init__(self, style_state: CharacterStyleState = None, min_quotes: int = 5, 
                 control_mode: str = None, q_info: pd.DataFrame = None):
        self.style_state = style_state if style_state is not None else CharacterStyleState()
        self.min_quotes = min_quotes
        self.control_mode = control_mode
        self.shuffle_map = {}
        self.other_quotes_pool = {}
        self.stats = []  # For fingerprint_statistics.csv
        
        # Runtime stats tracking
        self.durations = []
        
        if control_mode == "identity_shuffle" and q_info is not None:
            speakers = [s for s in q_info['speaker'].dropna().unique() if str(s).strip() != ""]
            shuffled = speakers.copy()
            random.seed(42)
            random.shuffle(shuffled)
            self.shuffle_map = dict(zip(speakers, shuffled))
            
        elif control_mode == "frequency_shuffle" and q_info is not None:
            explicit_df = q_info[q_info['quoteType'] == 'Explicit']
            speakers = explicit_df['speaker'].dropna().unique()
            for s in speakers:
                s_str = str(s).strip()
                others_df = explicit_df[explicit_df['speaker'] != s]
                self.other_quotes_pool[s_str] = others_df['quoteText'].dropna().tolist()
                
    def extract_features(self, quote_text: str, candidates: list[str], 
                         quote_id: str = None, quote_type: str = None, gold_speaker: str = None) -> dict[str, float]:
        """
        Returns a dictionary mapping candidate name -> similarity score.
        """
        start_time = time.perf_counter()
        
        if self.control_mode == "identity_shuffle":
            mapped_candidates = [self.shuffle_map.get(c, c) for c in candidates]
            scores = compute_similarity_scores(quote_text, mapped_candidates, self.style_state, self.min_quotes)
            mapped_back_scores = {}
            for original, shuffled in zip(candidates, mapped_candidates):
                mapped_back_scores[original] = scores[shuffled]
            final_scores = mapped_back_scores
        else:
            final_scores = compute_similarity_scores(quote_text, candidates, self.style_state, self.min_quotes)
            
        duration = time.perf_counter() - start_time
        self.durations.append(duration)
        
        # Log stats if metadata is provided
        if quote_id is not None:
            for c in candidates:
                lookup_cand = self.shuffle_map.get(c, c) if self.control_mode == "identity_shuffle" else c
                
                if lookup_cand not in self.style_state.state.fingerprints:
                    reason = "CANDIDATE_UNSEEN"
                elif self.style_state.state.fingerprints[lookup_cand].quotes_seen == 0:
                    reason = "NO_HISTORY"
                elif self.style_state.state.fingerprints[lookup_cand].quotes_seen < self.min_quotes:
                    reason = "BELOW_MIN_QUOTES"
                else:
                    reason = "AVAILABLE"
                    
                self.stats.append({
                    "quote_id": quote_id,
                    "quote_type": quote_type,
                    "candidate": c,
                    "similarity_score": final_scores.get(c, 0.0),
                    "reason_unavailable": reason,
                    "is_gold": 1 if c == gold_speaker else 0
                })
                
        return final_scores
        
    def get_memory_bytes(self) -> int:
        """
        Calculates the approximate memory usage of the StyleState in bytes.
        """
        total_size = sys.getsizeof(self.style_state.state.fingerprints)
        for k, fp in self.style_state.state.fingerprints.items():
            total_size += sys.getsizeof(k)
            total_size += sys.getsizeof(fp)
            total_size += sys.getsizeof(fp.character_id)
            total_size += sys.getsizeof(fp.texts)
            for t in fp.texts:
                total_size += sys.getsizeof(t)
        return total_size

    def get_largest_fingerprint_size(self) -> int:
        """
        Returns the maximum number of quotes stored in any character fingerprint.
        """
        if not self.style_state.state.fingerprints:
            return 0
        return max(fp.quotes_seen for fp in self.style_state.state.fingerprints.values())
        
    def update_state(self, speaker: str, quote_text: str):
        """
        Updates the character fingerprint with a new quote.
        """
        speaker = str(speaker).strip()
        if self.control_mode == "identity_shuffle":
            shuffled_speaker = self.shuffle_map.get(speaker, speaker)
            self.style_state.update(shuffled_speaker, quote_text)
        elif self.control_mode == "frequency_shuffle":
            pool = self.other_quotes_pool.get(speaker, [])
            if pool:
                random_quote = random.choice(pool)
                self.style_state.update(speaker, random_quote)
            else:
                self.style_state.update(speaker, quote_text)
        else:
            self.style_state.update(speaker, quote_text)
