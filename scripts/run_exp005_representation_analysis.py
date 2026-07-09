import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir
from src.evaluation.metrics import calculate_calibration_metrics
from sklearn.preprocessing import StandardScaler
from src.models.classical_models import PointwiseLogisticRanker

setup_logging()
logger = get_logger("exp005_representation_analysis")

def run_representation_analysis():
    logger.info("Starting EXP005: Representation Analysis...")
    
    input_file = get_data_dir() / "phase2" / "candidate_features.csv"
    if not input_file.exists():
        logger.error("Dataset not found.")
        return
        
    df = pd.read_csv(input_file)
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    # We will exclude symbolic features as we already learned they add noise/multicollinearity
    all_features = [c for c in df.columns if c not in ["quote_id", "novel", "candidate", "gold_speaker", "split", "label"] and not c.startswith("symbolic_")]
    
    X_train_raw = train_df[all_features]
    y_train = train_df['label']
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=all_features)
    
    logger.info("Running statsmodels Logistic Regression for Coefficient Analysis...")
    
    # Statsmodels Logit
    X_sm = sm.add_constant(X_train_scaled)
    model = sm.Logit(y_train.values, X_sm)
    result = model.fit(disp=False)
    
    # Extract stats
    stats_df = pd.DataFrame({
        "Coefficient": result.params,
        "Std Error": result.bse,
        "z-value": result.tvalues,
        "p-value": result.pvalues,
        "CI Lower": result.conf_int()[0],
        "CI Upper": result.conf_int()[1]
    })
    
    # Compute odds ratios
    stats_df["Odds Ratio"] = np.exp(stats_df["Coefficient"])
    
    # Drop intercept for the report and sort by absolute coefficient
    stats_df = stats_df.drop('const', errors='ignore')
    stats_df["AbsCoef"] = stats_df["Coefficient"].abs()
    stats_df = stats_df.sort_values(by="AbsCoef", ascending=False)
    stats_df = stats_df.drop("AbsCoef", axis=1)
    
    EXP_DIR = get_reports_dir() / "EXP005"
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Evaluate Calibration
    logger.info("Evaluating Calibration...")
    ranker = PointwiseLogisticRanker(random_state=42)
    ranker.fit(train_df[all_features], train_df['label'])
    y_prob_test = ranker.predict_proba(test_df[all_features])
    
    cal_metrics = calculate_calibration_metrics(test_df['label'], y_prob_test)
    
    # Calibration Plot
    logger.info("Plotting Reliability Diagram...")
    plt.figure(figsize=(6, 6))
    n_bins = 10
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob_test, bins) - 1
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accs = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (binids == i)
        if mask.sum() > 0:
            bin_accs[i] = test_df['label'].values[mask].mean()
        else:
            bin_accs[i] = np.nan
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.plot(bin_centers, bin_accs, 's-', label='Logistic Model')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram (Calibration Curve)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(EXP_DIR / "reliability_diagram.png")
    plt.close()
    
    # Representation Sufficiency Curve (Forward Selection)
    logger.info("Building Representation Sufficiency Curve...")
    
    ordered_features = stats_df.index.tolist()
    current_features = []
    sufficiency_results = []
    
    for i, feature in enumerate(ordered_features):
        current_features.append(feature)
        logger.info(f"Training with Top {i+1} features...")
        
        step_ranker = PointwiseLogisticRanker(random_state=42, max_iter=2000)
        step_ranker.fit(train_df[current_features], train_df['label'])
        acc = step_ranker.evaluate_ranking(test_df)['accuracy']
        
        sufficiency_results.append({
            "Num_Features": i+1,
            "Added_Feature": feature,
            "Accuracy": acc
        })
    
    suff_df = pd.DataFrame(sufficiency_results)
    
    # Plot Sufficiency Curve
    plt.figure(figsize=(10, 6))
    plt.plot(suff_df['Num_Features'], suff_df['Accuracy'], 'o-')
    plt.xlabel('Number of Features')
    plt.ylabel('Ranking Accuracy')
    plt.title('Representation Sufficiency Curve')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(suff_df['Num_Features'])
    plt.tight_layout()
    plt.savefig(EXP_DIR / "sufficiency_curve.png")
    plt.close()
    
    # Single-Feature Ablations (for top 5 candidate features)
    logger.info("Running Single-Feature Ablations on top features...")
    baseline_acc = suff_df['Accuracy'].iloc[-1]
    ablation_results = []
    for feature in ordered_features[:5]:
        feats = [f for f in all_features if f != feature]
        step_ranker = PointwiseLogisticRanker(random_state=42)
        step_ranker.fit(train_df[feats], train_df['label'])
        acc = step_ranker.evaluate_ranking(test_df)['accuracy']
        ablation_results.append({
            "Ablated_Feature": feature,
            "Accuracy": acc,
            "Drop": baseline_acc - acc
        })
        
    ablation_df = pd.DataFrame(ablation_results)
    
    # Write Report
    report_file = EXP_DIR / "representation_analysis_report.md"
    with open(report_file, 'w') as f:
        f.write("# EXP005: Representation Analysis Report\n\n")
        
        f.write("## 1. Coefficient Analysis\n")
        f.write("Features sorted by absolute standardized coefficient magnitude. This highlights their independent importance.\n\n")
        f.write("| Feature | Coefficient | Std Error | 95% CI | p-value | Odds Ratio |\n")
        f.write("|---------|-------------|-----------|--------|---------|------------|\n")
        for idx, row in stats_df.iterrows():
            f.write(f"| {idx} | {row['Coefficient']:.4f} | {row['Std Error']:.4f} | [{row['CI Lower']:.4f}, {row['CI Upper']:.4f}] | {row['p-value']:.4e} | {row['Odds Ratio']:.4f} |\n")
            
        f.write("\n## 2. Representation Sufficiency (Forward Selection)\n")
        f.write("How much representation is actually necessary? We start with the best feature and incrementally add the next best.\n\n")
        f.write("![Representation Sufficiency Curve](/home/Aditya/speaker-attribution-research/results/EXP005/sufficiency_curve.png)\n\n")
        f.write("| Features Used | Latest Feature Added | Ranking Accuracy |\n")
        f.write("|---------------|----------------------|------------------|\n")
        for idx, row in suff_df.iterrows():
            f.write(f"| {row['Num_Features']} | {row['Added_Feature']} | {row['Accuracy']*100:.2f}% |\n")
            
        f.write("\n## 3. Single-Feature Ablation\n")
        f.write(f"Baseline Accuracy (All features): {baseline_acc*100:.2f}%\n\n")
        f.write("| Ablated Feature | New Accuracy | Absolute Drop |\n")
        f.write("|-----------------|--------------|---------------|\n")
        for idx, row in ablation_df.iterrows():
            f.write(f"| {row['Ablated_Feature']} | {row['Accuracy']*100:.2f}% | {row['Drop']*100:.2f}% |\n")
            
        f.write("\n## 4. Probability Calibration\n")
        f.write("Since this is a ranking task, probability calibration is critical.\n\n")
        f.write("![Reliability Diagram](/home/Aditya/speaker-attribution-research/results/EXP005/reliability_diagram.png)\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| ROC-AUC | {cal_metrics['roc_auc']:.4f} |\n")
        f.write(f"| PR-AUC | {cal_metrics['pr_auc']:.4f} |\n")
        f.write(f"| Brier Score | {cal_metrics['brier_score']:.4f} |\n")
        f.write(f"| Expected Calibration Error (ECE) | {cal_metrics['ece']:.4f} |\n")
        
    logger.info(f"Report saved to {report_file}")

if __name__ == "__main__":
    run_representation_analysis()
