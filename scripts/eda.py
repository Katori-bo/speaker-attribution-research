import json
import os
from collections import Counter
import numpy as np
from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir

setup_logging()
logger = get_logger("eda")

DATA_DIR = get_data_dir()
REPORT_PATH = get_reports_dir() / "eda_report.md"

def run_eda():
    logger.info("Starting exploratory data analysis...")
    if not DATA_DIR.exists():
        logger.error(f"Dataset directory not found: {DATA_DIR}")
        return

    quotes_file = DATA_DIR / "quotes.json"
    coref_file = DATA_DIR / "coref.json"
    
    # We will compute these metrics
    num_books = 0
    num_dialogues = 0
    speakers = set()
    quote_lengths = []
    speaker_counts = Counter()
    longest_dialogue_length = 0
    num_unknown_speakers = 0
    
    # Check if files exist before processing to avoid crashes
    if quotes_file.exists():
        logger.info(f"Loading {quotes_file}")
        try:
            with open(quotes_file, 'r') as f:
                quotes_data = json.load(f)
                
            num_books = len(quotes_data) if isinstance(quotes_data, dict) else 1
            
            # Simple assumption of structure for the EDA script:
            # quotes_data could be a list of quotes or dict grouped by book.
            quotes = quotes_data if isinstance(quotes_data, list) else []
            if isinstance(quotes_data, dict):
                for book, book_quotes in quotes_data.items():
                    quotes.extend(book_quotes)
                    
            num_dialogues = len(quotes)
            
            for q in quotes:
                speaker = q.get("speaker", "Unknown")
                if speaker == "Unknown":
                    num_unknown_speakers += 1
                speakers.add(speaker)
                speaker_counts[speaker] += 1
                
                text = q.get("text", "")
                length = len(text.split())
                quote_lengths.append(length)
                if length > longest_dialogue_length:
                    longest_dialogue_length = length
                    
        except json.JSONDecodeError:
            logger.error(f"Failed to parse {quotes_file}")
    else:
        logger.warning(f"{quotes_file} not found. EDA metrics will be 0.")

    avg_quote_length = np.mean(quote_lengths) if quote_lengths else 0
    num_speakers = len(speakers)
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating EDA report at {REPORT_PATH}")
    
    with open(REPORT_PATH, 'w') as f:
        f.write("# Exploratory Data Analysis Report\n\n")
        f.write("## Dataset Statistics\n")
        f.write(f"- **Number of books:** {num_books}\n")
        f.write(f"- **Number of dialogues (quotes):** {num_dialogues}\n")
        f.write(f"- **Number of unique speakers:** {num_speakers}\n")
        f.write(f"- **Number of unknown speakers:** {num_unknown_speakers}\n")
        f.write(f"- **Average quote length (words):** {avg_quote_length:.2f}\n")
        f.write(f"- **Longest dialogue (words):** {longest_dialogue_length}\n\n")
        
        f.write("## Top 10 Speakers\n")
        for speaker, count in speaker_counts.most_common(10):
            f.write(f"- {speaker}: {count}\n")
            
    logger.info("EDA completed successfully.")

if __name__ == "__main__":
    run_eda()
