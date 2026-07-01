import pytest
from src.evaluation.metrics import calculate_all_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def test_metrics_match_sklearn():
    y_true = [0, 1, 0]
    y_pred = [0, 1, 1]
    
    metrics = calculate_all_metrics(y_true, y_pred, average='weighted')
    
    assert metrics["accuracy"] == accuracy_score(y_true, y_pred)
    assert metrics["precision"] == precision_score(y_true, y_pred, average='weighted', zero_division=0)
    assert metrics["recall"] == recall_score(y_true, y_pred, average='weighted', zero_division=0)
    assert metrics["f1"] == f1_score(y_true, y_pred, average='weighted', zero_division=0)
