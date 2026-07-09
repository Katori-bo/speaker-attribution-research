import pytest
import pandas as pd
from src.evaluation.metrics import (
    compute_anchor_state_drift_stats,
    compute_post_anchor_window_accuracy,
    compute_confidence_at_error_inclusive,
    compute_cascade_survivorship
)

def test_anchor_state_drift_detection():
    # synthetic sequence where persisted state != gold prev
    df = pd.DataFrame([
        {
            "quote_id": "test_1",
            "score": 0.9,
            "label": 1,
            "is_anchor_fired": 1,
            "state_drifted": True,
            "persisted_last_speaker": "WrongSpeaker",
            "gold_prev_speaker": "RightSpeaker",
            "anchor_attributed_speaker": "RightSpeaker",
            "state_reset_applied": True
        }
    ])
    
    stats = compute_anchor_state_drift_stats(df)
    assert stats["anchor_events"] == 1
    assert stats["state_drifted_at_anchor"] == 1
    assert stats["state_correct_at_anchor"] == 0
    assert stats["state_resets_applied"] == 1
    assert stats["drifted_that_would_reset"] == 1

def test_post_anchor_window_stratified():
    # verify implicit-only filtering and K-window slicing
    ear_df = pd.DataFrame([
        {"novel": "A", "quote_id": "A_0", "score": 0.9, "label": 1, "is_anchor_fired": 1, "state_drifted": True},
        {"novel": "A", "quote_id": "A_1", "score": 0.9, "label": 1},
        {"novel": "A", "quote_id": "A_2", "score": 0.9, "label": 0}
    ])
    
    fa_df = pd.DataFrame([
        {"quote_id": "A_0", "score": 0.9, "label": 0},
        {"quote_id": "A_1", "score": 0.9, "label": 0},
        {"quote_id": "A_2", "score": 0.9, "label": 0}
    ])
    
    qt_map = {
        "A_0": "Explicit Named",
        "A_1": "Implicit",
        "A_2": "Implicit"
    }
    
    res = compute_post_anchor_window_accuracy(ear_df, fa_df, qt_map, k=[2])
    
    # window 2 should include A_1 and A_2 for anchor at A_0
    assert len(res) == 2
    assert res.iloc[0]["quote_id"] == "A_1"
    assert res.iloc[0]["ear_correct"] == 1
    assert res.iloc[0]["fa_correct"] == 0
    assert res.iloc[0]["state_drifted"] == True
    
    assert res.iloc[1]["quote_id"] == "A_2"
    assert res.iloc[1]["ear_correct"] == 0
    
def test_confidence_at_error_inclusive():
    # verify unrecovered errors are counted
    df = pd.DataFrame([
        {"quote_id": "Q_1", "score": 0.8, "label": 0, "novel": "A"}, # Error
        {"quote_id": "Q_2", "score": 0.9, "label": 0, "novel": "A"}, # Error
        {"quote_id": "Q_3", "score": 0.7, "label": 1, "novel": "A"}  # Correct
    ])
    
    res = compute_confidence_at_error_inclusive(df)
    assert res.iloc[0]["Total_Errors"] == 2
    assert res.iloc[0]["Mean_Confidence_At_Error_Inclusive"] == "0.8500"

def test_cascade_survivorship_audit():
    df = pd.DataFrame([
        {"quote_id": "Q_0", "score": 0.9, "label": 1, "novel": "A"},
        {"quote_id": "Q_1", "score": 0.8, "label": 0, "novel": "A"}, # Error cascade starts
        {"quote_id": "Q_2", "score": 0.9, "label": 0, "novel": "A"}, 
        {"quote_id": "Q_3", "score": 0.9, "label": 0, "novel": "A"}, 
        {"quote_id": "Q_4", "score": 0.9, "label": 0, "novel": "A"}, 
        {"quote_id": "Q_5", "score": 0.9, "label": 0, "novel": "A"}, 
        {"quote_id": "Q_6", "score": 0.9, "label": 0, "novel": "A"}, # 6th wrong quote -> never recovers within 5
        {"quote_id": "Q_7", "score": 0.9, "label": 1, "novel": "A"}, # Recovers on 7th
        
        {"quote_id": "Q_8", "score": 0.8, "label": 0, "novel": "A"}, # 2nd error cascade starts
        {"quote_id": "Q_9", "score": 0.9, "label": 0, "novel": "A"},
        {"quote_id": "Q_10", "score": 0.9, "label": 1, "novel": "A"}, # Recovers within 5 (dist=2)
    ])
    
    stats = compute_cascade_survivorship(df)
    assert stats["total_errors"] == 2
    assert stats["recovered_within_5"] == 1
    assert stats["unrecovered"] == 1
    assert stats["pct_excluded_from_drift_table"] == "50.0%"
