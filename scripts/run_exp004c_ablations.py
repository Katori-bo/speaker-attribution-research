import pandas as pd
from pathlib import Path
from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir
from src.models.classical_models import PointwiseLogisticRanker

setup_logging()
logger = get_logger("exp004c_ablations")

def run_ablations():
    logger.info("Starting EXP004C: Feature Family Ablations...")
    
    input_file = get_data_dir() / "phase2" / "candidate_features.csv"
    if not input_file.exists():
        logger.error(f"Dataset not found at {input_file}.")
        return
        
    df = pd.read_csv(input_file)
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    all_features = [c for c in df.columns if c not in ["quote_id", "novel", "candidate", "gold_speaker", "split", "label"]]
    
    families = {
        "Lexical": [f for f in all_features if f.startswith('lexical_')],
        "Candidate": [f for f in all_features if f.startswith('candidate_')],
        "Discourse": [f for f in all_features if f.startswith('discourse_')],
        "Conversation": [f for f in all_features if f.startswith('conversation_')],
        "Symbolic": [f for f in all_features if f.startswith('symbolic_')]
    }
    
    logger.info(f"Training Baseline (All Features)...")
    base_ranker = PointwiseLogisticRanker(random_state=42)
    base_ranker.fit(train_df[all_features], train_df['label'])
    base_acc = base_ranker.evaluate_ranking(test_df)['accuracy']
    
    results = []
    results.append({"Ablation": "None (All Features)", "Accuracy": base_acc, "Drop": 0.0})
    
    for family, family_features in families.items():
        if not family_features:
            continue
            
        logger.info(f"Ablating {family} features...")
        features_to_use = [f for f in all_features if f not in family_features]
        
        ranker = PointwiseLogisticRanker(random_state=42)
        ranker.fit(train_df[features_to_use], train_df['label'])
        acc = ranker.evaluate_ranking(test_df)['accuracy']
        drop = base_acc - acc
        
        results.append({"Ablation": f"Removed {family}", "Accuracy": acc, "Drop": drop})
        
    EXP_DIR = get_reports_dir() / "EXP004C"
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sort by drop (largest drop first)
    sorted_results = sorted(results[1:], key=lambda x: x["Drop"], reverse=True)
    
    with open(EXP_DIR / "ablation_report.md", 'w') as f:
        f.write("# EXP004C: Feature Family Ablation Report\n\n")
        f.write(f"- **Baseline Accuracy:** {base_acc*100:.2f}%\n\n")
        f.write("## Ablation Results\n")
        f.write("| Ablated Family | New Accuracy | Absolute Drop |\n")
        f.write("|----------------|--------------|---------------|\n")
        for res in sorted_results:
            f.write(f"| {res['Ablation']} | {res['Accuracy']*100:.2f}% | {res['Drop']*100:.2f}% |\n")
            
    logger.info(f"Ablation Report saved to {EXP_DIR / 'ablation_report.md'}")

if __name__ == "__main__":
    run_ablations()
