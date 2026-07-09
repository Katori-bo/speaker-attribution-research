import pytest
import pandas as pd
from pathlib import Path
from src.evaluation.beam_search_runner import run_beam_search_evaluation
from src.evaluation.runner import run_evaluation, load_frozen_exp014_dataset, train_exp014_model
from src.evaluation.discourse_mode import FullyAutoregressiveMode

@pytest.mark.timeout(300)
def test_beam_search_k1_equivalence():
    """
    Ensures that Beam Search with K=1 produces identical predictions to 
    the Fully Autoregressive (greedy) mode.
    """
    df = load_frozen_exp014_dataset()
    model, feature_names = train_exp014_model(df)
    
    # Subsample to a single novel for test speed
    test_novels = df[df['split'] == 'test']['novel'].unique()
    target_novel = test_novels[0]
    
    small_df = df[(df['novel'] == target_novel) | (df['split'] == 'train')].copy()
    
    # 1. Run FA Mode
    fa_mode = FullyAutoregressiveMode()
    fa_preds = run_evaluation(fa_mode, small_df, model, feature_names)
    idx_fa = fa_preds.groupby('quote_id', sort=False)['score'].idxmax()
    fa_pred_speakers = fa_preds.loc[idx_fa]['candidate'].values
    
    # 2. Run Beam Search K=1
    beam_preds, oracle_stats, oracle_logs_df = run_beam_search_evaluation(small_df, model, feature_names, beam_size=1)
    # Beam search already returns the final branch of predictions chosen
    beam_pred_speakers = beam_preds['candidate'].values
    
    # 3. Assert exact equivalence
    assert len(fa_pred_speakers) == len(beam_pred_speakers), f"Length mismatch: {len(fa_pred_speakers)} vs {len(beam_pred_speakers)}"
    assert (fa_pred_speakers == beam_pred_speakers).all(), "K=1 Beam Search differs from Fully Autoregressive baseline!"

from unittest.mock import patch
import numpy as np

@patch("src.evaluation.beam_search_runner.pd.read_csv")
@patch("src.evaluation.beam_search_runner.extract_dynamic_features")
def test_oracle_death_reason(mock_extract, mock_read_csv):
    # Mock extract_dynamic_features to do nothing
    mock_extract.return_value = {}
    
    # We will test two novels:
    # Novel 1: Quote 0 -> Gold candidate exists but pruned
    # Novel 2: Quote 0 -> Gold candidate absent
    
    q_info_novel_1 = pd.DataFrame({
        "quote_id": ["n1_q0"],
        "speaker": ["Gold_1"],
        "mentionEntitiesList": ["[]"],
        "addressees": ["[]"],
        "quoteByteSpans": ["[[10, 20]]"]
    })
    
    q_info_novel_2 = pd.DataFrame({
        "quote_id": ["n2_q0"],
        "speaker": ["Gold_2"],
        "mentionEntitiesList": ["[]"],
        "addressees": ["[]"],
        "quoteByteSpans": ["[[10, 20]]"]
    })
    
    def side_effect_read_csv(filepath, **kwargs):
        if "novel_1" in str(filepath):
            return q_info_novel_1
        elif "novel_2" in str(filepath):
            return q_info_novel_2
        return pd.DataFrame()
        
    mock_read_csv.side_effect = side_effect_read_csv
    
    df = pd.DataFrame({
        "split": ["test", "test", "test", "test"],
        "novel": ["novel_1", "novel_1", "novel_1", "novel_2"],
        "quote_id": ["n1_q0", "n1_q0", "n1_q0", "n2_q0"],
        "candidate": ["Other_1", "Other_2", "Gold_1", "Other_3"],
        "gold_speaker": ["Gold_1", "Gold_1", "Gold_1", "Gold_2"],
        "discourse_dialogue_position": [1.0, 1.0, 1.0, 1.0],
        "quote_start_byte": [10, 10, 10, 10],
        "dummy_feat": [1, 1, 1, 1]
    })
    
    class FakeModel:
        def predict_proba(self, X):
            if len(X) == 3:
                # For novel_1, Gold_1 is the 3rd candidate. We give it score 0.1, others 0.8 and 0.9.
                return np.array([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1]])
            else:
                # For novel_2, doesn't matter, gold is missing
                return np.array([[0.1, 0.9]])
                
    model = FakeModel()
    feature_names = ["dummy_feat"]
    
    # Beam size = 1, so Gold_1 (score 0.1) will be pruned.
    pred_df, oracle_stats, oracle_logs_df = run_beam_search_evaluation(df, model, feature_names, beam_size=1)
    
    log_n1 = oracle_logs_df[oracle_logs_df['novel'] == 'novel_1'].iloc[0]
    log_n2 = oracle_logs_df[oracle_logs_df['novel'] == 'novel_2'].iloc[0]
    
    assert log_n1['death_reason'] == "beam_pruned"
    assert log_n2['death_reason'] == "candidate_missing"

