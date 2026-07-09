import json
import pandas as pd
from pathlib import Path
from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir
from src.models.classical_models import PointwiseLogisticRanker

setup_logging()
logger = get_logger("exp004b_logistic_regression")

def run_logistic_regression():
    logger.info("Starting EXP004B: Logistic Regression Evaluation...")
    
    input_file = get_data_dir() / "phase2" / "candidate_features.csv"
    if not input_file.exists():
        logger.error(f"Dataset not found at {input_file}. Run generate_dataset_p2.py first.")
        return
        
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} candidate rows.")
    
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    features = [c for c in df.columns if c not in ["quote_id", "novel", "candidate", "gold_speaker", "split", "label"]]
    
    X_train = train_df[features]
    y_train = train_df['label']
    
    logger.info(f"Training Logistic Regression Ranker on {len(X_train)} instances with {len(features)} features...")
    ranker = PointwiseLogisticRanker(random_state=42)
    ranker.fit(X_train, y_train)
    
    logger.info("Evaluating on Training Set...")
    train_results = ranker.evaluate_ranking(train_df)
    logger.info(f"Train Accuracy: {train_results['accuracy']*100:.2f}%")
    
    logger.info("Evaluating on Test Set...")
    test_results = ranker.evaluate_ranking(test_df)
    logger.info(f"Test Accuracy: {test_results['accuracy']*100:.2f}%")
    
    # Feature Importance
    importance = ranker.get_feature_importance()
    sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    
    EXP_DIR = get_reports_dir() / "EXP004B"
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(EXP_DIR / "logistic_regression_report.md", 'w') as f:
        f.write("# EXP004B: Logistic Regression Baseline Report\n\n")
        f.write(f"- **Train Accuracy (Solvable Quotes):** {train_results['accuracy']*100:.2f}%\n")
        f.write(f"- **Test Accuracy (Solvable Quotes):** {test_results['accuracy']*100:.2f}%\n\n")
        
        f.write("## Feature Coefficients (Absolute Importance)\n")
        f.write("| Feature | Coefficient |\n")
        f.write("|---------|-------------|\n")
        for feature, coef in sorted_importance:
            f.write(f"| {feature} | {coef:.4f} |\n")
            
    logger.info(f"Report saved to {EXP_DIR / 'logistic_regression_report.md'}")

if __name__ == "__main__":
    run_logistic_regression()
