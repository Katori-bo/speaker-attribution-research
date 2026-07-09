import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List

from src.neural.schemas import CandidateExample, QuoteStep, NovelSequence

class SpeakerSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        """
        Input: existing EXP014 dataframe
        Output: list[NovelSequence]
        """
        self.sequences = []
        self._build_sequences(df, feature_cols)

    def _build_sequences(self, df: pd.DataFrame, feature_cols: List[str]):
        for novel_id, novel_df in df.groupby('novel'):
            def get_quote_idx(q_id):
                try:
                    return int(str(q_id).split('_')[-1])
                except:
                    return 0

            quotes = []
            # Sort quote IDs first to maintain chronological order
            unique_quote_ids = sorted(novel_df['quote_id'].unique(), key=get_quote_idx)
            
            for quote_id in unique_quote_ids:
                quote_df = novel_df[novel_df['quote_id'] == quote_id]
                
                candidates = []
                for _, row in quote_df.iterrows():
                    feat_vec = torch.tensor([row[c] for c in feature_cols], dtype=torch.float32)
                    is_gold = bool(row['label'])
                    
                    candidates.append(CandidateExample(
                        candidate_id=row['candidate'],
                        features=feat_vec,
                        is_gold=is_gold
                    ))
                
                gold_speaker = quote_df.iloc[0]['gold_speaker']
                
                quotes.append(QuoteStep(
                    quote_id=quote_id,
                    candidates=candidates,
                    gold_speaker=gold_speaker
                ))
            
            self.sequences.append(NovelSequence(
                novel_id=novel_id,
                quotes=quotes
            ))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> NovelSequence:
        return self.sequences[idx]
