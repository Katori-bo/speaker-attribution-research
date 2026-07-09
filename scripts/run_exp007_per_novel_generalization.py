import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score

from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir
from src.models.classical_models import PointwiseLogisticRanker

setup_logging()
logger = get_logger("exp007_per_novel_generalization")

def run_per_novel_generalization():
    logger.info("Starting EXP007: Per-Novel Generalization...")
    
    input_file = get_data_dir() / "phase2" / "candidate_features.csv"
    df = pd.read_csv(input_file)
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    all_features = [c for c in df.columns if c not in ["quote_id", "novel", "candidate", "gold_speaker", "split", "label"] and not c.startswith("symbolic_")]
    top_3 = ['candidate_is_recent_mention', 'candidate_is_previous_speaker', 'candidate_is_explicit_mention']
    
    # Train the models on the whole train set
    ranker_top3 = PointwiseLogisticRanker(random_state=42)
    ranker_top3.fit(train_df[top_3], train_df['label'])
    
    ranker_all = PointwiseLogisticRanker(random_state=42)
    ranker_all.fit(train_df[all_features], train_df['label'])
    
    # We will evaluate on BOTH train and test split for this analysis just to have enough data per novel
    # Some novels might entirely be in train or test. Let's evaluate per novel on all quotes from that novel.
    
    novel_results = []
    
    for novel, group in df.groupby('novel'):
        # Evaluate Top 3
        acc_top3 = ranker_top3.evaluate_ranking(group)['accuracy']
        
        # Evaluate All
        acc_all = ranker_all.evaluate_ranking(group)['accuracy']
        
        # Metadata
        num_speakers = group[group['label'] == 1]['gold_speaker'].nunique()
        candidate_set_size = group.groupby('quote_id')['candidate'].count().mean()
        avg_dialogue_length = group['lexical_quote_length_chars'].mean() # approximate
        
        novel_results.append({
            "Novel": novel,
            "Accuracy_Top3": acc_top3,
            "Accuracy_All": acc_all,
            "d_Accuracy": acc_all - acc_top3,
            "Num_Speakers": num_speakers,
            "Avg_Candidate_Set_Size": candidate_set_size,
            "Avg_Quote_Length": avg_dialogue_length
        })
        
    res_df = pd.DataFrame(novel_results)
    
    EXP_DIR = get_reports_dir() / "EXP007"
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(res_df['Num_Speakers'], res_df['d_Accuracy'] * 100, alpha=0.7, s=res_df['Avg_Candidate_Set_Size']*10)
    
    for i, row in res_df.iterrows():
        # Annotate novels with significant accuracy differences
        if abs(row['d_Accuracy']) > 0.02:
            plt.annotate(row['Novel'], (row['Num_Speakers'], row['d_Accuracy'] * 100))
            
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Number of Unique Speakers in Novel')
    plt.ylabel('Δ Accuracy (All Features - Top 3) %')
    plt.title('Does Cast Size Demand Richer Features?')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(EXP_DIR / "novel_cast_vs_features.png")
    plt.close()
    
    report_file = EXP_DIR / "per_novel_generalization_report.md"
    with open(report_file, 'w') as f:
        f.write("# EXP007: Per-Novel Generalization Report\n\n")
        f.write("Does the Top 3 feature representation suffice for all novels, or do complex novels demand richer context?\n\n")
        f.write("![Cast vs Features](/home/Aditya/speaker-attribution-research/results/EXP007/novel_cast_vs_features.png)\n\n")
        
        res_df = res_df.sort_values(by="d_Accuracy", ascending=False)
        f.write(res_df.to_markdown(index=False))
        
    logger.info(f"Report saved to {report_file}")

if __name__ == "__main__":
    run_per_novel_generalization()
