import csv
import ast
import hashlib
from pathlib import Path
from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir
from src.baseline.candidate_generator import CandidateGenerator
from src.discourse.discourse_state import MinimalDiscourseState
from src.features.extractor import FeatureExtractor

setup_logging()
logger = get_logger("generate_dataset_p2")

def parse_stringified_list(val):
    try:
        return ast.literal_eval(val)
    except:
        return []

def flatten_mentions(mentions):
    flat = []
    if isinstance(mentions, list):
        for item in mentions:
            if isinstance(item, list):
                flat.extend(flatten_mentions(item))
            else:
                flat.append(item)
    return flat

def get_split(novel_name: str) -> str:
    # Deterministic split based on novel name hash
    hash_val = int(hashlib.md5(novel_name.encode('utf-8')).hexdigest(), 16)
    return "test" if hash_val % 5 == 0 else "train"

def generate_dataset():
    logger.info("Starting Phase 2 Dataset Generation...")
    
    pdnc_dir = get_data_dir() / "data"
    generator = CandidateGenerator()
    extractor = FeatureExtractor()
    state = MinimalDiscourseState()
    
    novels = sorted([d for d in pdnc_dir.iterdir() if d.is_dir()])
    
    dataset_rows = []
    feature_names = set()
    
    for novel in novels:
        quote_file = novel / "quotation_info.csv"
        if not quote_file.exists(): continue
        
        split = get_split(novel.name)
        state.reset_conversation()
        previous_speakers = []
        
        with open(quote_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                gold_speaker = row.get("speaker", "Unknown").strip()
                quote_text = row.get("quoteText", "")
                context_text = row.get("referringExpression", "").strip()
                quote_id = row.get("quote_id", f"{novel.name}_{i}")
                
                mentions_raw = parse_stringified_list(row.get("mentionEntitiesList", "[]"))
                addressees_raw = parse_stringified_list(row.get("addressees", "[]"))
                explicit_mentions = flatten_mentions(mentions_raw) + flatten_mentions(addressees_raw)
                
                candidates = set()
                for m in explicit_mentions:
                    if m: candidates.add(m)
                for s in set(previous_speakers[-15:]):
                    if s: candidates.add(s)
                
                state.update(
                    previous_speakers[-1] if previous_speakers else None, 
                    explicit_mentions, 
                    candidates
                )
                
                quote_dict = {
                    'quote_text': quote_text,
                    'context_text': context_text
                }
                
                quote_spans_raw = parse_stringified_list(row.get("quoteByteSpans", "[]"))
                start_byte = -1
                end_byte = -1
                if quote_spans_raw and len(quote_spans_raw) > 0:
                    try:
                        start_byte = int(quote_spans_raw[0][0])
                        end_byte = int(quote_spans_raw[-1][1])
                    except (IndexError, ValueError, TypeError):
                        pass

                for candidate in candidates:
                    features = extractor.extract(quote_dict, candidate, state)
                    feature_names.update(features.keys())
                    
                    row_data = {
                        "quote_id": quote_id,
                        "novel": novel.name,
                        "candidate": candidate,
                        "gold_speaker": gold_speaker,
                        "split": split,
                        "label": 1 if candidate == gold_speaker else 0,
                        "quote_start_byte": start_byte,
                        "quote_end_byte": end_byte
                    }
                    row_data.update(features)
                    dataset_rows.append(row_data)
                
                if gold_speaker != "Unknown":
                    previous_speakers.append(gold_speaker)
                    
    # Save dataset
    output_dir = get_data_dir() / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "candidate_features.csv"
    
    # Sort feature names for deterministic output
    sorted_features = sorted(list(feature_names))
    fieldnames = ["quote_id", "novel", "candidate", "gold_speaker", "split", "label", "quote_start_byte", "quote_end_byte"] + sorted_features
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset_rows)
        
    logger.info(f"Dataset generated with {len(dataset_rows)} candidate rows across {len(sorted_features)} features.")
    logger.info(f"Saved to {output_file}")

if __name__ == "__main__":
    generate_dataset()
