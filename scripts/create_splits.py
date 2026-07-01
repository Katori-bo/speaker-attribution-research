import json
from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_splits_dir

setup_logging()
logger = get_logger("create_splits")

SPLITS_DIR = get_splits_dir()
DATA_DIR = get_data_dir()

def create_splits():
    logger.info("Creating train/val/test splits at novel level...")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Placeholder split logic, this will be implemented properly when data structure is confirmed
    train_split = {"novels": ["novel1", "novel2"]}
    val_split = {"novels": ["novel3"]}
    test_split = {"novels": ["novel4"]}
    
    with open(SPLITS_DIR / "train.json", "w") as f:
        json.dump(train_split, f, indent=2)
    with open(SPLITS_DIR / "validation.json", "w") as f:
        json.dump(val_split, f, indent=2)
    with open(SPLITS_DIR / "test.json", "w") as f:
        json.dump(test_split, f, indent=2)
        
    logger.info("Splits created successfully in data/splits/")

if __name__ == "__main__":
    create_splits()
