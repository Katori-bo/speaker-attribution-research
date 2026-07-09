import json
import csv
import ast
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir
from src.experiment.recorder import ExperimentRecorder
from src.baseline.candidate_generator import CandidateGenerator

setup_logging()
logger = get_logger("exp002")

DATA_DIR = get_data_dir()
EXP_DIR = get_reports_dir() / "EXP002"
METRICS_PATH = EXP_DIR / "candidate_generation_metrics.json"
REPORT_PATH = EXP_DIR / "candidate_generation_report.md"
VIS_DIR = EXP_DIR / "visualizations"

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

def run_exp002():
    logger.info("Starting EXP002 Candidate Generation evaluation...")
    
    with ExperimentRecorder("EXP002"):
        pdnc_dir = DATA_DIR / "data"
        if not pdnc_dir.exists():
            raise FileNotFoundError(f"PDNC dataset missing at {pdnc_dir}. Cannot proceed.")
            
        generator = CandidateGenerator()
        
        total_quotes = 0
        hits = 0
        set_sizes = []
        
        novel_stats = defaultdict(lambda: {"quotes": 0, "hits": 0})
        speaker_frequency = Counter()
        speaker_hits = defaultdict(int)
        
        novels = [d for d in pdnc_dir.iterdir() if d.is_dir()]
        logger.info(f"Evaluating candidates across {len(novels)} novels...")
        
        for novel in novels:
            novel_name = novel.name
            quote_file = novel / "quotation_info.csv"
            if not quote_file.exists():
                continue
                
            previous_speakers = []
            
            try:
                with open(quote_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        gold_speaker = row.get("speaker", "Unknown").strip()
                        speaker_frequency[gold_speaker] += 1
                        
                        # Parse mentions
                        mentions_raw = parse_stringified_list(row.get("mentionEntitiesList", "[]"))
                        addressees_raw = parse_stringified_list(row.get("addressees", "[]"))
                        
                        explicit_mentions = flatten_mentions(mentions_raw) + flatten_mentions(addressees_raw)
                        
                        # Generate candidates
                        candidates = generator.generate_candidates(
                            explicit_mentions=explicit_mentions,
                            nearby_characters=[], # No easy access without full text parsing
                            previous_participants=list(set(previous_speakers[-5:])), # Last 5 speakers
                            local_paragraph_mentions=[]
                        )
                        
                        total_quotes += 1
                        set_sizes.append(len(candidates))
                        novel_stats[novel_name]["quotes"] += 1
                        
                        # Check hit (Oracle Accuracy)
                        is_hit = gold_speaker in candidates
                        if is_hit:
                            hits += 1
                            novel_stats[novel_name]["hits"] += 1
                            speaker_hits[gold_speaker] += 1
                            
                        # Update history
                        if gold_speaker != "Unknown":
                            previous_speakers.append(gold_speaker)
                            
            except Exception as e:
                logger.error(f"Error parsing {novel_name}: {e}")
                
        # Calculate metrics
        overall_recall = hits / total_quotes if total_quotes else 0
        avg_size = np.mean(set_sizes) if set_sizes else 0
        med_size = np.median(set_sizes) if set_sizes else 0
        p95_size = np.percentile(set_sizes, 95) if set_sizes else 0
        max_size = np.max(set_sizes) if set_sizes else 0
        
        # Speaker Frequency Bucketing (Rare vs Frequent)
        rare_quotes = 0
        rare_hits = 0
        freq_quotes = 0
        freq_hits = 0
        for speaker, count in speaker_frequency.items():
            if count <= 5:
                rare_quotes += count
                rare_hits += speaker_hits[speaker]
            else:
                freq_quotes += count
                freq_hits += speaker_hits[speaker]
                
        rare_recall = rare_hits / rare_quotes if rare_quotes else 0
        freq_recall = freq_hits / freq_quotes if freq_quotes else 0
        
        # Save Outputs
        EXP_DIR.mkdir(parents=True, exist_ok=True)
        VIS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(set_sizes, bins=range(max(set_sizes)+2), color='lightcoral', edgecolor='black', align='left')
        plt.title('Candidate Set Size Distribution')
        plt.xlabel('Number of Candidates')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(VIS_DIR / "candidate_set_sizes.png")
        plt.close()
        
        metrics = {
            "total_quotes": total_quotes,
            "candidate_recall": float(overall_recall),
            "oracle_accuracy": float(overall_recall),
            "avg_candidate_set_size": float(avg_size),
            "median_candidate_set_size": float(med_size),
            "p95_candidate_set_size": float(p95_size),
            "max_candidate_set_size": int(max_size),
            "rare_speaker_recall": float(rare_recall),
            "frequent_speaker_recall": float(freq_recall),
            "novel_recall": {k: float(v["hits"] / v["quotes"]) for k, v in novel_stats.items() if v["quotes"] > 0}
        }
        
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        with open(REPORT_PATH, 'w') as f:
            f.write("# EXP002 Candidate Generation Report\n\n")
            f.write("## Global Metrics\n")
            f.write(f"- **Candidate Recall (Oracle Accuracy):** {overall_recall*100:.2f}%\n")
            f.write(f"- **Average Candidate Set Size:** {avg_size:.2f}\n")
            f.write(f"- **Median Candidate Set Size:** {med_size:.2f}\n")
            f.write(f"- **95th Percentile Size:** {p95_size:.2f}\n")
            f.write(f"- **Max Set Size:** {max_size}\n\n")
            
            f.write("## Recall by Speaker Frequency\n")
            f.write(f"- **Rare Speakers (<= 5 quotes):** {rare_recall*100:.2f}%\n")
            f.write(f"- **Frequent Speakers (> 5 quotes):** {freq_recall*100:.2f}%\n\n")
            
            f.write("## Recall by Novel\n")
            f.write("| Novel | Quotes | Recall |\n")
            f.write("|-------|--------|--------|\n")
            for novel_name, ns in sorted(novel_stats.items()):
                n_recall = ns["hits"] / ns["quotes"] if ns["quotes"] > 0 else 0
                f.write(f"| {novel_name} | {ns['quotes']} | {n_recall*100:.2f}% |\n")
            
        logger.info(f"EXP002 metrics saved to {METRICS_PATH}")
        logger.info("EXP002 completed successfully.")

if __name__ == "__main__":
    run_exp002()
