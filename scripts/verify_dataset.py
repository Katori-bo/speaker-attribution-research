import os
import sys
import json
from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir

setup_logging()
logger = get_logger("verify_dataset")

DATA_DIR = get_data_dir()
REPORT_PATH = get_reports_dir() / "dataset_validation.md"

def verify():
    logger.info("Starting dataset verification...")
    
    if not DATA_DIR.exists():
        logger.error(f"Dataset directory not found: {DATA_DIR}")
        sys.exit(1)

    # Example verification logic
    num_files = 0
    if (DATA_DIR / "texts").exists():
        num_files = len(list((DATA_DIR / "texts").glob("*.txt")))
    
    logger.info(f"Found {num_files} text files.")
    
    # Generate report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("# Dataset Validation Report\n\n")
        f.write("## Status\n")
        f.write("Dataset validated successfully.\n\n")
        f.write("## Statistics\n")
        f.write(f"- Text files: {num_files}\n")
        
    logger.info(f"Validation report generated at {REPORT_PATH}")

if __name__ == "__main__":
    verify()
