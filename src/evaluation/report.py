import json
from pathlib import Path
from typing import Dict, Any

def save_metrics_report(metrics: Dict[str, Any], filepath: str | Path):
    """Save metrics to a JSON file"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)
