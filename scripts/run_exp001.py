import json
import csv
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir
from src.experiment.recorder import ExperimentRecorder

setup_logging()
logger = get_logger("exp001")

DATA_DIR = get_data_dir()
EXP_DIR = get_reports_dir() / "EXP001"
REPORT_PATH = EXP_DIR / "dataset_report.md"
STATS_PATH = EXP_DIR / "dataset_statistics.json"
VIS_DIR = EXP_DIR / "visualizations"

def plot_length_distribution(lengths, filepath):
    if not lengths: return
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title('Quote Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(filepath)
    plt.close()

def plot_speaker_frequency(speaker_counts, filepath):
    if not speaker_counts: return
    top_speakers = speaker_counts.most_common(20)
    names, counts = zip(*top_speakers)
    plt.figure(figsize=(12, 6))
    plt.bar(names, counts, color='lightgreen')
    plt.title('Top 20 Speakers by Frequency')
    plt.xlabel('Speaker')
    plt.ylabel('Number of Quotes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def run_exp001():
    logger.info("Starting EXP001 Dataset Characterization...")
    
    with ExperimentRecorder("EXP001"):
        pdnc_dir = DATA_DIR / "data"
        if not pdnc_dir.exists():
            raise FileNotFoundError(f"PDNC dataset missing at {pdnc_dir}. Cannot proceed.")
            
        num_novels = 0
        num_chapters = 0
        num_quotes = 0
        speakers = set()
        quote_lengths = []
        speaker_counts = Counter()
        longest_quote_length = 0
        num_unknown_speakers = 0
        novel_stats = {}
        
        novels = [d for d in pdnc_dir.iterdir() if d.is_dir()]
        num_novels = len(novels)
        
        logger.info(f"Loading {num_novels} novels from {pdnc_dir}")
        
        for novel in novels:
            novel_name = novel.name
            novel_stats[novel_name] = {"quotes": 0, "speakers": set()}
            quote_file = novel / "quotation_info.csv"
            if not quote_file.exists():
                logger.warning(f"Missing quotation_info.csv in {novel.name}")
                continue
                
            try:
                with open(quote_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        num_quotes += 1
                        
                        speaker = row.get("speaker", "Unknown").strip()
                        if speaker == "" or speaker.lower() == "unknown":
                            speaker = "Unknown"
                            num_unknown_speakers += 1
                            
                        speakers.add(speaker)
                        speaker_counts[speaker] += 1
                        novel_stats[novel_name]["quotes"] += 1
                        novel_stats[novel_name]["speakers"].add(speaker)
                        
                        text = row.get("quoteText", "")
                        length = len(text.split())
                        quote_lengths.append(length)
                        if length > longest_quote_length:
                            longest_quote_length = length
                            
            except Exception as e:
                raise RuntimeError(f"Failed to parse {quote_file}") from e

        avg_quote_length = np.mean(quote_lengths) if quote_lengths else 0
        num_speakers = len(speakers)
        
        # Save Outputs
        EXP_DIR.mkdir(parents=True, exist_ok=True)
        VIS_DIR.mkdir(parents=True, exist_ok=True)
        
        plot_length_distribution(quote_lengths, VIS_DIR / "quote_length_distribution.png")
        plot_speaker_frequency(speaker_counts, VIS_DIR / "top_speakers.png")
        
        # Long tail
        speakers_less_than_5 = sum(1 for c in speaker_counts.values() if c < 5)
        long_tail_pct = (speakers_less_than_5 / num_speakers) * 100 if num_speakers else 0
        
        stats = {
            "num_novels": num_novels,
            "num_chapters": num_chapters,
            "num_quotes": num_quotes,
            "num_speakers": num_speakers,
            "num_unknown_speakers": num_unknown_speakers,
            "avg_quote_length": float(avg_quote_length),
            "longest_quote_length": longest_quote_length,
            "long_tail_pct": float(long_tail_pct),
            "novel_breakdown": {k: {"quotes": v["quotes"], "speakers": len(v["speakers"])} for k, v in novel_stats.items()}
        }
        
        with open(STATS_PATH, 'w') as f:
            json.dump(stats, f, indent=4)
            
        logger.info(f"Statistics saved to {STATS_PATH}")
        
        with open(REPORT_PATH, 'w') as f:
            f.write("# EXP001 Dataset Characterization Report\n\n")
            f.write("## Dataset Statistics\n")
            f.write(f"- **Number of novels:** {num_novels}\n")
            f.write(f"- **Number of chapters:** {num_chapters} (Not directly extracted in basic EDA)\n")
            f.write(f"- **Number of quotes:** {num_quotes}\n")
            f.write(f"- **Number of unique speakers:** {num_speakers}\n")
            f.write(f"- **Number of unknown speakers:** {num_unknown_speakers}\n")
            f.write(f"- **Average quote length (words):** {avg_quote_length:.2f}\n")
            f.write(f"- **Longest quote length (words):** {longest_quote_length}\n")
            f.write(f"- **Long-tail speakers (< 5 quotes):** {long_tail_pct:.2f}%\n\n")
            
            f.write("## Per Novel Breakdown\n")
            f.write("| Novel | Quotes | Speakers |\n")
            f.write("|-------|--------|----------|\n")
            for novel_name, ns in sorted(novel_stats.items()):
                f.write(f"| {novel_name} | {ns['quotes']} | {len(ns['speakers'])} |\n")
            
            f.write("\n## Top 20 Speakers\n")
            for speaker, count in speaker_counts.most_common(20):
                f.write(f"- {speaker}: {count}\n")
                
            f.write("\n## Visualizations\n")
            f.write("Visualizations are saved in the `visualizations/` directory.\n")
                
        logger.info(f"EDA Report saved to {REPORT_PATH}")
        logger.info("EXP001 completed successfully.")

if __name__ == "__main__":
    run_exp001()
