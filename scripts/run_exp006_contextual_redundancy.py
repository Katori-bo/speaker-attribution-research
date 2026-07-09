import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import matthews_corrcoef, log_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from pathlib import Path

from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir
from src.evaluation.metrics import expected_calibration_error
from src.models.classical_models import PointwiseLogisticRanker

setup_logging()
logger = get_logger("exp006_contextual_redundancy")

def get_feature_types(df, features):
    binary = []
    continuous = []
    for f in features:
        if df[f].nunique() == 2 and set(df[f].dropna().unique()) == {0, 1}:
            binary.append(f)
        else:
            continuous.append(f)
    return binary, continuous

def experiment_a_correlation(df, features, out_dir):
    logger.info("Running Experiment A: Correlation Structure...")
    binary_feats, cont_feats = get_feature_types(df, features)
    
    # We will compute a custom matrix
    corr_matrix = pd.DataFrame(index=features, columns=features, dtype=float)
    
    for f1 in features:
        for f2 in features:
            if f1 == f2:
                corr_matrix.loc[f1, f2] = 1.0
                continue
            
            if f1 in binary_feats and f2 in binary_feats:
                # Phi coefficient (Matthews correlation)
                try:
                    corr = matthews_corrcoef(df[f1], df[f2])
                except:
                    corr = np.nan
            elif (f1 in binary_feats and f2 in cont_feats) or (f1 in cont_feats and f2 in binary_feats):
                # Point-biserial
                bin_f = f1 if f1 in binary_feats else f2
                cont_f = f2 if f1 in binary_feats else f1
                try:
                    corr, _ = stats.pointbiserialr(df[bin_f], df[cont_f])
                except:
                    corr = np.nan
            else:
                # Pearson
                try:
                    corr, _ = stats.pearsonr(df[f1], df[f2])
                except:
                    corr = np.nan
                    
            corr_matrix.loc[f1, f2] = corr
            
    corr_matrix.to_csv(out_dir / "correlation_matrix.csv")
    return corr_matrix

def experiment_b_conditional(train_df, test_df, top_features, remaining_features):
    logger.info("Running Experiment B: Conditional Importance...")
    results = []
    
    # Baseline (Top 3)
    ranker = PointwiseLogisticRanker(random_state=42)
    ranker.fit(train_df[top_features], train_df['label'])
    base_acc = ranker.evaluate_ranking(test_df)['accuracy']
    
    base_probs = ranker.predict_proba(test_df[top_features])
    base_roc = roc_auc_score(test_df['label'], base_probs)
    base_loss = log_loss(test_df['label'], base_probs)
    base_ece = expected_calibration_error(test_df['label'], base_probs)
    
    results.append({
        "Added_Feature": "None (Baseline)",
        "Accuracy": base_acc,
        "ROC-AUC": base_roc,
        "LogLoss": base_loss,
        "ECE": base_ece
    })
    
    for feat in remaining_features:
        feats_to_use = top_features + [feat]
        r = PointwiseLogisticRanker(random_state=42)
        r.fit(train_df[feats_to_use], train_df['label'])
        
        acc = r.evaluate_ranking(test_df)['accuracy']
        probs = r.predict_proba(test_df[feats_to_use])
        
        try:
            roc = roc_auc_score(test_df['label'], probs)
        except:
            roc = np.nan
            
        loss = log_loss(test_df['label'], probs)
        ece = expected_calibration_error(test_df['label'], probs)
        
        results.append({
            "Added_Feature": feat,
            "Accuracy": acc,
            "d_Accuracy": acc - base_acc,
            "ROC-AUC": roc,
            "d_ROC-AUC": roc - base_roc,
            "LogLoss": loss,
            "d_LogLoss": loss - base_loss,
            "ECE": ece,
            "d_ECE": ece - base_ece
        })
        
    return pd.DataFrame(results)

def experiment_c_residual(train_df, test_df, top_features):
    logger.info("Running Experiment C: Residual Error Taxonomy...")
    ranker = PointwiseLogisticRanker(random_state=42)
    ranker.fit(train_df[top_features], train_df['label'])
    
    scores = ranker.predict_proba(test_df[top_features])
    test_eval = test_df.copy()
    test_eval['score'] = scores
    
    errors = []
    
    for quote_id, group in test_eval.groupby('quote_id'):
        if group['label'].sum() > 0:
            best_idx = group['score'].idxmax()
            best_candidate = group.loc[best_idx, 'candidate']
            gold_speaker = group.loc[best_idx, 'gold_speaker']
            
            if best_candidate != gold_speaker:
                # This is an error. Collect characteristics
                # Get the actual correct candidate row
                gold_row = group[group['label'] == 1].iloc[0]
                pred_row = group.loc[best_idx]
                
                error_type = "Unknown"
                if gold_row['candidate_is_recent_mention'] == 0 and gold_row['candidate_is_previous_speaker'] == 0 and gold_row['candidate_is_explicit_mention'] == 0:
                    error_type = "Gold has no explicit signals"
                elif gold_row['candidate_is_recent_mention'] == 1 and pred_row['candidate_is_previous_speaker'] == 1:
                    error_type = "Confused Mention for Previous Speaker"
                elif gold_row['discourse_context_length'] > 20:
                    error_type = "Long Context Narration"
                    
                errors.append({
                    "quote_id": quote_id,
                    "novel": gold_row['novel'],
                    "gold_speaker": gold_speaker,
                    "predicted_speaker": best_candidate,
                    "error_type": error_type,
                    "dialogue_position": gold_row['discourse_dialogue_position']
                })
                
    err_df = pd.DataFrame(errors)
    if not err_df.empty:
        summary = err_df['error_type'].value_counts().reset_index()
        summary.columns = ['Error Type', 'Count']
        return summary
    return pd.DataFrame(columns=['Error Type', 'Count'])

def experiment_d_interactions(train_df, test_df, top_features):
    logger.info("Running Experiment D: Interaction Testing...")
    # Hypothesis driven interactions
    interactions = [
        ("candidate_is_previous_speaker", "discourse_context_length"),
        ("candidate_is_previous_speaker", "conversation_length"),
        ("candidate_is_explicit_mention", "discourse_context_length")
    ]
    
    # Baseline
    r_base = PointwiseLogisticRanker(random_state=42)
    r_base.fit(train_df[top_features], train_df['label'])
    base_acc = r_base.evaluate_ranking(test_df)['accuracy']
    
    results = []
    
    for f1, f2 in interactions:
        t_df = train_df.copy()
        v_df = test_df.copy()
        
        inter_name = f"{f1}_X_{f2}"
        t_df[inter_name] = t_df[f1] * t_df[f2]
        v_df[inter_name] = v_df[f1] * v_df[f2]
        
        feats = top_features + [inter_name]
        r = PointwiseLogisticRanker(random_state=42)
        r.fit(t_df[feats], t_df['label'])
        acc = r.evaluate_ranking(v_df)['accuracy']
        
        results.append({
            "Interaction": inter_name,
            "Accuracy": acc,
            "d_Accuracy": acc - base_acc
        })
        
    return pd.DataFrame(results)

def experiment_e_stability(train_df, features):
    logger.info("Running Experiment E: Stability Analysis...")
    X = train_df[features]
    y = train_df['label']
    
    # Sample down for speed since this is just a CV check
    if len(X) > 20000:
        sample_idx = np.random.choice(len(X), 20000, replace=False)
        X = X.iloc[sample_idx]
        y = y.iloc[sample_idx]
        
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    
    sfs_forward = SequentialFeatureSelector(clf, n_features_to_select=3, direction='forward', cv=3)
    sfs_forward.fit(X, y)
    forward_features = [features[i] for i, selected in enumerate(sfs_forward.get_support()) if selected]
    
    sfs_backward = SequentialFeatureSelector(clf, n_features_to_select=3, direction='backward', cv=3)
    sfs_backward.fit(X, y)
    backward_features = [features[i] for i, selected in enumerate(sfs_backward.get_support()) if selected]
    
    return forward_features, backward_features

def run_redundancy_analysis():
    logger.info("Starting EXP006: Contextual Redundancy Analysis...")
    
    input_file = get_data_dir() / "phase2" / "candidate_features.csv"
    df = pd.read_csv(input_file)
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    all_features = [c for c in df.columns if c not in ["quote_id", "novel", "candidate", "gold_speaker", "split", "label"] and not c.startswith("symbolic_")]
    
    top_3 = ['candidate_is_recent_mention', 'candidate_is_previous_speaker', 'candidate_is_explicit_mention']
    remaining = [f for f in all_features if f not in top_3]
    
    EXP_DIR = get_reports_dir() / "EXP006"
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    corr_df = experiment_a_correlation(train_df, all_features, EXP_DIR)
    cond_df = experiment_b_conditional(train_df, test_df, top_3, remaining)
    err_df = experiment_c_residual(train_df, test_df, top_3)
    inter_df = experiment_d_interactions(train_df, test_df, top_3)
    fwd_feats, bwd_feats = experiment_e_stability(train_df, all_features)
    
    # Write Report
    report_file = EXP_DIR / "redundancy_analysis_report.md"
    with open(report_file, 'w') as f:
        f.write("# EXP006: Contextual Redundancy Analysis Report\n\n")
        
        f.write("## Experiment A: Correlation Structure\n")
        f.write("Full matrix saved to `correlation_matrix.csv`. High correlations indicate redundancy prior to modeling.\n\n")
        
        f.write("## Experiment B: Conditional Importance\n")
        f.write("If the Top 3 are present, does adding a 4th feature improve probabilities or accuracy?\n\n")
        f.write(cond_df.to_markdown(index=False) + "\n\n")
        
        f.write("## Experiment C: Residual Error Taxonomy\n")
        f.write("What kind of errors remain when only using the Top 3 features?\n\n")
        f.write(err_df.to_markdown(index=False) + "\n\n")
        
        f.write("## Experiment D: Interaction Testing\n")
        f.write("Do hypothesis-driven feature interactions provide complementary signal?\n\n")
        f.write(inter_df.to_markdown(index=False) + "\n\n")
        
        f.write("## Experiment E: Stability Analysis\n")
        f.write("Does feature selection converge to the Top-3 set regardless of search strategy (3-fold CV)?\n\n")
        f.write(f"- **Forward Selection Top 3:** {fwd_feats}\n")
        f.write(f"- **Backward Elimination Top 3:** {bwd_feats}\n")
        
    logger.info(f"Report saved to {report_file}")

if __name__ == "__main__":
    run_redundancy_analysis()
