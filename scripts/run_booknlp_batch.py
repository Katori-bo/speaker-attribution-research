import os
import json
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime
import pkg_resources

try:
    from booknlp.booknlp import BookNLP
except ImportError:
    BookNLP = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_booknlp_version():
    try:
        return pkg_resources.get_distribution("booknlp").version
    except pkg_resources.DistributionNotFound:
        return "unknown"

def main():
    if BookNLP is None:
        logger.error("BookNLP is not installed. Please install it in the virtual environment.")
        return

    data_dir = Path("data/raw/pdnc/data")
    output_dir = Path("data/raw/pdnc/booknlp_out")
    status_file = output_dir / "processing_status.json"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load status
    status = {}
    if status_file.exists():
        with open(status_file, "r") as f:
            status = json.load(f)
            
    # Initialize BookNLP
    pipeline_config = {
        "pipeline": "entity,quote,supersense,event,coref", 
        "model": "small"
    }
    logger.info("Initializing BookNLP pipeline...")
    booknlp = BookNLP("en", pipeline_config)
    booknlp_version = get_booknlp_version()
    
    novels = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(novels)} novels.")
    
    for novel in novels:
        if status.get(novel) == "complete":
            logger.info(f"Skipping {novel} (already complete)")
            continue
            
        novel_txt = data_dir / novel / "novel_text.txt"
        if not novel_txt.exists():
            logger.warning(f"No novel_text.txt found for {novel}")
            continue
            
        logger.info(f"Processing {novel}...")
        
        novel_out_dir = output_dir / novel
        novel_out_dir.mkdir(exist_ok=True)
        
        start_time = time.time()
        
        try:
            # Process using BookNLP
            booknlp.process(str(novel_txt), str(novel_out_dir), novel)
            
            runtime = time.time() - start_time
            
            # Count metrics
            tokens_file = novel_out_dir / f"{novel}.tokens"
            entities_file = novel_out_dir / f"{novel}.entities"
            quotes_file = novel_out_dir / f"{novel}.quotes"
            book_file = novel_out_dir / f"{novel}.book"
            
            tokens_count = 0
            if tokens_file.exists():
                with open(tokens_file, "r") as f:
                    tokens_count = sum(1 for line in f) - 1 # header
            
            entities_count = 0
            if entities_file.exists():
                with open(entities_file, "r") as f:
                    entities_count = sum(1 for line in f) - 1 # header
                    
            mentions_count = 0
            chains_count = 0
            if book_file.exists():
                with open(book_file, "r") as f:
                    book_data = json.load(f)
                    chains_count = len(book_data.get("characters", []))
                    for char in book_data.get("characters", []):
                        mentions_count += len(char.get("mentions", {}))
            
            # Write metadata
            metadata = {
                "novel": novel,
                "booknlp_version": booknlp_version,
                "pipeline_config": pipeline_config,
                "processing_timestamp": datetime.utcnow().isoformat() + "Z",
                "counts": {
                    "tokens": max(0, tokens_count),
                    "entities": max(0, entities_count),
                    "mentions": mentions_count,
                    "chains": chains_count
                },
                "runtime_seconds": round(runtime, 2)
            }
            
            with open(novel_out_dir / "booknlp_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
            status[novel] = "complete"
            
        except Exception as e:
            logger.error(f"Failed to process {novel}: {str(e)}")
            logger.error(traceback.format_exc())
            status[novel] = "failed"
            
        # Checkpoint status
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)

        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
