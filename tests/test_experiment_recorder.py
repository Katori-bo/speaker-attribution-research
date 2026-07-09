import pytest
import json
from pathlib import Path
from src.experiment.recorder import ExperimentRecorder

def test_experiment_recorder_success(tmp_path, monkeypatch):
    monkeypatch.setattr("src.experiment.recorder.get_reports_dir", lambda: tmp_path / "reports")
    
    with ExperimentRecorder("TEST001", config_path="nonexistent.yaml"):
        # simulate some work
        pass
        
    metadata_file = tmp_path / "reports" / "TEST001" / "metadata.json"
    assert metadata_file.exists()
    
    with open(metadata_file) as f:
        metadata = json.load(f)
        
    assert metadata["status"] == "COMPLETE"
    assert "execution_time_seconds" in metadata

def test_experiment_recorder_fail_fast(tmp_path, monkeypatch):
    monkeypatch.setattr("src.experiment.recorder.get_reports_dir", lambda: tmp_path / "reports")
    
    with pytest.raises(FileNotFoundError):
        with ExperimentRecorder("TEST002", config_path="nonexistent.yaml"):
            raise FileNotFoundError("Simulated dataset missing")
            
    metadata_file = tmp_path / "reports" / "TEST002" / "metadata.json"
    assert metadata_file.exists()
    
    with open(metadata_file) as f:
        metadata = json.load(f)
        
    assert metadata["status"] == "FAILED"
    assert metadata["reason"] == "FileNotFoundError"
