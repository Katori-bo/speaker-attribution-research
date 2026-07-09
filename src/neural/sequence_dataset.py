import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset

@dataclass
class NovelSequenceTensor:
    novel_id: str
    quote_ids: List[str]
    candidate_features: torch.Tensor  # [num_quotes, max_candidates, feature_dim]
    candidate_mask: torch.BoolTensor  # [num_quotes, max_candidates]
    candidate_ids: torch.LongTensor   # [num_quotes, max_candidates]
    gold_index: torch.LongTensor      # [num_quotes]
    gold_speaker_id: torch.LongTensor # [num_quotes]

def build_character_vocab(df: pd.DataFrame, vocab_path: str = "data/character_vocab.json") -> Dict[str, int]:
    candidates = set(df['candidate'].dropna().unique())
    golds = set(df['gold_speaker'].dropna().unique())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    
    df_copy = df.copy()
    df_copy['novel_candidate'] = df_copy['novel'] + "::" + df_copy['candidate'].astype(str)
    df_copy['novel_gold'] = df_copy['novel'] + "::" + df_copy['gold_speaker'].astype(str)
    
    cands = set(df_copy['novel_candidate'].dropna().unique())
    golds = set(df_copy['novel_gold'].dropna().unique())
    unique_candidates = sorted(cands | golds)
    
    for c in unique_candidates:
        vocab[c] = len(vocab)
            
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=4)
        
    return vocab

class TensorSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], feature_mode: str = 'full', vocab: Dict[str, int] = None, scaler=None):
        """
        feature_mode: 'full' or 'state_free'
        """
        self.feature_mode = feature_mode
        self.vocab = vocab or {}
        
        self.mutable_discourse_features = [
            'candidate_is_last_speaker',
            'candidate_is_previous_speaker',
            'candidate_in_participant_stack',
            'candidate_stack_depth',
            'conversation_speaker_change',
            'conv_active_id',
            'conv_interruption_distance'
        ]
        
        if self.feature_mode == 'state_free':
            self.active_features = [c for c in feature_cols if c not in self.mutable_discourse_features]
        else:
            self.active_features = feature_cols
            
        self.sequences = []
        self._build_sequences(df, scaler)
        
    def _get_char_id(self, novel_id: str, char_str: str) -> int:
        full_id = f"{novel_id}::{char_str}"
        return self.vocab.get(full_id, self.vocab.get("<UNK>", 1))
        
    def _build_sequences(self, df: pd.DataFrame, scaler=None):
        global_max_candidates = df.groupby('quote_id').size().max()
        
        for novel_id, novel_df in df.groupby('novel'):
            def get_quote_idx(q_id):
                try: return int(str(q_id).split('_')[-1])
                except: return 0
                
            unique_quote_ids = sorted(novel_df['quote_id'].unique(), key=get_quote_idx)
            num_quotes = len(unique_quote_ids)
            
            if global_max_candidates == 0:
                continue
                
            candidate_features = torch.zeros(num_quotes, global_max_candidates, len(self.active_features), dtype=torch.float32)
            candidate_mask = torch.zeros(num_quotes, global_max_candidates, dtype=torch.bool)
            candidate_ids = torch.zeros(num_quotes, global_max_candidates, dtype=torch.long)
            gold_index = torch.zeros(num_quotes, dtype=torch.long)
            gold_speaker_id = torch.zeros(num_quotes, dtype=torch.long)
            
            # Scale features if a scaler is provided
            features_df = novel_df[self.active_features].values
            if scaler is not None:
                features_df = scaler.transform(features_df)
                
            # Dictionary mapping quote_id -> subset of scaled features
            novel_df_scaled = novel_df.copy()
            for i, col in enumerate(self.active_features):
                novel_df_scaled[col] = features_df[:, i]
                
            for q_idx, quote_id in enumerate(unique_quote_ids):
                quote_df = novel_df_scaled[novel_df_scaled['quote_id'] == quote_id]
                
                gold_speaker = quote_df.iloc[0]['gold_speaker']
                
                gold_idx_found = False
                for c_idx, (_, row) in enumerate(quote_df.iterrows()):
                    feats = row[self.active_features].values.astype(np.float32)
                    candidate_features[q_idx, c_idx, :] = torch.tensor(feats)
                    candidate_mask[q_idx, c_idx] = True
                    candidate_ids[q_idx, c_idx] = self._get_char_id(novel_id, row['candidate'])
                    
                    if str(row['candidate']) == str(gold_speaker):
                        gold_index[q_idx] = c_idx
                        gold_speaker_id[q_idx] = self._get_char_id(novel_id, gold_speaker)
                        gold_idx_found = True
                        
                # If no gold candidate is found (or label is absent), default to 0.
                if not gold_idx_found:
                    gold_index[q_idx] = 0
                    
            self.sequences.append(NovelSequenceTensor(
                novel_id=novel_id,
                quote_ids=unique_quote_ids,
                candidate_features=candidate_features,
                candidate_mask=candidate_mask,
                candidate_ids=candidate_ids,
                gold_index=gold_index,
                gold_speaker_id=gold_speaker_id
            ))
            
    def __len__(self) -> int:
        return len(self.sequences)
        
    def __getitem__(self, idx: int) -> NovelSequenceTensor:
        return self.sequences[idx]
