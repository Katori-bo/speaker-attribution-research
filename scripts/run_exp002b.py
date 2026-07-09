import json
import csv
import ast
import time
from pathlib import Path
from collections import Counter, defaultdict
from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir

setup_logging()
logger = get_logger("exp002b")

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

def load_aliases(novel_dir):
    alias_map = {}
    char_file = novel_dir / "character_info.csv"
    if char_file.exists():
        with open(char_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                main_name = row.get("Main Name", "").strip()
                if not main_name: continue
                
                alias_map[main_name.lower()] = main_name
                
                aliases_raw = row.get("Aliases", "{}")
                try:
                    aliases = ast.literal_eval(aliases_raw) if aliases_raw.startswith("{") else []
                    for alias in aliases:
                        alias_map[alias.lower()] = main_name
                except:
                    pass
    return alias_map

def run_variant(novels, variant_name, use_aliases=False, window_size=5):
    logger.info(f"Running Variant: {variant_name} (Aliases={use_aliases}, Window={window_size})")
    
    start_time = time.time()
    
    total_quotes = 0
    hits = 0
    set_sizes = []
    
    speaker_frequency = Counter()
    speaker_hits = defaultdict(int)
    
    for novel in novels:
        quote_file = novel / "quotation_info.csv"
        if not quote_file.exists(): continue
            
        alias_map = load_aliases(novel) if use_aliases else {}
        previous_speakers = []
        
        with open(quote_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gold_speaker = row.get("speaker", "Unknown").strip()
                speaker_frequency[gold_speaker] += 1
                
                mentions_raw = parse_stringified_list(row.get("mentionEntitiesList", "[]"))
                addressees_raw = parse_stringified_list(row.get("addressees", "[]"))
                explicit_mentions = flatten_mentions(mentions_raw) + flatten_mentions(addressees_raw)
                
                # Base candidates
                candidates = set()
                for m in explicit_mentions:
                    if m: candidates.add(m)
                for s in set(previous_speakers[-window_size:]):
                    if s: candidates.add(s)
                    
                # Apply alias normalization
                if use_aliases:
                    normalized_candidates = set()
                    for c in candidates:
                        normalized_candidates.add(c)
                        if c.lower() in alias_map:
                            normalized_candidates.add(alias_map[c.lower()])
                    candidates = normalized_candidates
                
                total_quotes += 1
                set_sizes.append(len(candidates))
                
                if gold_speaker in candidates:
                    hits += 1
                    speaker_hits[gold_speaker] += 1
                    
                if gold_speaker != "Unknown":
                    previous_speakers.append(gold_speaker)
                    
    runtime = time.time() - start_time
    recall = hits / total_quotes if total_quotes else 0
    avg_size = sum(set_sizes) / len(set_sizes) if set_sizes else 0
    efficiency = recall / avg_size if avg_size > 0 else 0
    
    # Buckets
    rare_quotes, rare_hits = 0, 0
    freq_quotes, freq_hits = 0, 0
    for s, c in speaker_frequency.items():
        if c <= 5:
            rare_quotes += c
            rare_hits += speaker_hits[s]
        else:
            freq_quotes += c
            freq_hits += speaker_hits[s]
            
    rare_recall = rare_hits / rare_quotes if rare_quotes else 0
    freq_recall = freq_hits / freq_quotes if freq_quotes else 0
    
    return {
        "variant": variant_name,
        "runtime_seconds": runtime,
        "recall": recall,
        "avg_set_size": avg_size,
        "efficiency": efficiency,
        "rare_recall": rare_recall,
        "freq_recall": freq_recall
    }

def run_exp002b():
    pdnc_dir = get_data_dir() / "data"
    novels = [d for d in pdnc_dir.iterdir() if d.is_dir()]
    
    variants = [
        ("V0_Baseline", False, 5),
        ("V1_Alias", True, 5),
        ("V2_Window15", False, 15),
        ("V3_Window30", False, 30),
        ("V4_Alias_Window15", True, 15)
    ]
    
    results = []
    for v_name, use_aliases, window in variants:
        res = run_variant(novels, v_name, use_aliases, window)
        results.append(res)
        
    EXP_DIR = get_reports_dir() / "EXP002b"
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(EXP_DIR / "ablation_results.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    with open(EXP_DIR / "ablation_report.md", 'w') as f:
        f.write("# EXP002b Candidate Generation Ablations\n\n")
        f.write("| Variant | Recall | Avg Size | Efficiency | Rare Recall | Freq Recall | Runtime (s) |\n")
        f.write("|---------|--------|----------|------------|-------------|-------------|-------------|\n")
        
        for r in results:
            f.write(f"| {r['variant']} | {r['recall']*100:.2f}% | {r['avg_set_size']:.2f} | {r['efficiency']*100:.2f}% | {r['rare_recall']*100:.2f}% | {r['freq_recall']*100:.2f}% | {r['runtime_seconds']:.2f} |\n")
            
    logger.info("EXP002b completed successfully.")

if __name__ == "__main__":
    run_exp002b()
