import pytest
import yaml
from pathlib import Path

def test_paths_config_exists():
    assert Path("config/paths.yaml").exists()

def test_paths_config_is_valid():
    with open("config/paths.yaml", "r") as f:
        config = yaml.safe_load(f)
    assert "raw_data" in config
    assert "splits" in config
    
def test_experiment_config_is_valid():
    with open("config/experiment.yaml", "r") as f:
        config = yaml.safe_load(f)
    assert "seed" in config
    assert "batch_size" in config
