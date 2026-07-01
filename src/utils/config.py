import yaml
from pathlib import Path

def load_config(config_path: str = "config/paths.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_data_dir() -> Path:
    config = load_config()
    return Path(config.get("raw_data", "data/raw/pdnc"))

def get_splits_dir() -> Path:
    config = load_config()
    return Path(config.get("splits", "data/splits"))

def get_reports_dir() -> Path:
    config = load_config()
    return Path(config.get("reports", "reports"))
