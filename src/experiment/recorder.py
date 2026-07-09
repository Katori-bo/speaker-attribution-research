import json
import time
import shutil
from pathlib import Path
from datetime import datetime, timezone
from src.experiment.metadata import gather_metadata
from src.utils.config import get_reports_dir

class ExperimentRecorder:
    """
    Context manager that tracks experiment execution, captures configuration,
    and reliably generates metadata.json with the experiment's final status.
    """
    def __init__(self, exp_id: str, config_path: str = "config/experiment.yaml"):
        self.exp_id = exp_id
        self.config_path = Path(config_path)
        self.exp_dir = get_reports_dir() / self.exp_id
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Snapshot configuration
        if self.config_path.exists():
            shutil.copy(self.config_path, self.exp_dir / "config_used.yaml")
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        execution_time = end_time - self.start_time
        
        metadata = gather_metadata()
        metadata["experiment_id"] = self.exp_id
        metadata["execution_time_seconds"] = round(execution_time, 4)
        metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        if exc_type is None:
            metadata["status"] = "COMPLETE"
        else:
            metadata["status"] = "FAILED"
            metadata["reason"] = exc_type.__name__
            
        with open(self.exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
            
        # Return False so exceptions are propagated (Fail Fast)
        return False
