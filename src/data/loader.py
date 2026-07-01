import json
from pathlib import Path
from typing import Iterator, Dict, Any
from src.utils.logger import get_logger
from src.utils.config import get_data_dir

logger = get_logger("data_loader")

class PDNCLoader:
    """
    Loads PDNC data from raw JSON files into Python objects.
    Produces an iterator of quotes.
    """
    def __init__(self, data_dir: str | Path = None):
        self.data_dir = Path(data_dir) if data_dir else get_data_dir()
        self.quotes_file = self.data_dir / "quotes.json"
        
    def load(self) -> Iterator[Dict[str, Any]]:
        """
        Yields parsed quotes one by one.
        """
        if not self.quotes_file.exists():
            logger.error(f"Quotes file not found: {self.quotes_file}")
            return
            
        logger.info(f"Loading quotes from {self.quotes_file}")
        try:
            with open(self.quotes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # If data is grouped by book
            if isinstance(data, dict):
                for book_id, quotes in data.items():
                    for quote in quotes:
                        quote['book_id'] = book_id
                        yield quote
            # If data is a flat list
            elif isinstance(data, list):
                for quote in data:
                    yield quote
            else:
                logger.error("Unexpected data format in quotes.json")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON in {self.quotes_file}: {e}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
