import csv
import random
import ast
from pathlib import Path
from src.utils.config import get_data_dir
from src.baseline.candidate_generator import CandidateGenerator

def parse_stringified_list(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
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

def sample_errors():
    pdnc_dir = get_data_dir() / "data"
    generator = CandidateGenerator()
    
    failures = []
    
    novels = [d for d in pdnc_dir.iterdir() if d.is_dir()]
    for novel in novels:
        quote_file = novel / "quotation_info.csv"
        if not quote_file.exists(): continue
        
        previous_speakers = []
        
        with open(quote_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gold_speaker = row.get("speaker", "Unknown").strip()
                
                mentions_raw = parse_stringified_list(row.get("mentionEntitiesList", "[]"))
                addressees_raw = parse_stringified_list(row.get("addressees", "[]"))
                explicit_mentions = flatten_mentions(mentions_raw) + flatten_mentions(addressees_raw)
                
                candidates = generator.generate_candidates(
                    explicit_mentions=explicit_mentions,
                    nearby_characters=[],
                    previous_participants=list(set(previous_speakers[-5:])),
                    local_paragraph_mentions=[]
                )
                
                if gold_speaker not in candidates:
                    failures.append({
                        "novel": novel.name,
                        "quote_id": row.get("quoteID"),
                        "gold_speaker": gold_speaker,
                        "quote_text": row.get("quoteText"),
                        "candidates": list(candidates),
                        "explicit_mentions": explicit_mentions,
                        "previous_participants": list(set(previous_speakers[-5:]))
                    })
                    
                if gold_speaker != "Unknown":
                    previous_speakers.append(gold_speaker)
                    
    random.seed(42)
    sample = random.sample(failures, min(100, len(failures)))
    
    out_dir = Path("results/EXP002b")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / "error_sample.csv"
    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["novel", "quote_id", "gold_speaker", "quote_text", "candidates", "explicit_mentions", "previous_participants"])
        writer.writeheader()
        writer.writerows(sample)
        
    print(f"Sampled {len(sample)} errors to {out_file}")

if __name__ == "__main__":
    sample_errors()
