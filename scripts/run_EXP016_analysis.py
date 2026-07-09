import os
import json
import pandas as pd
from src.utils.logger import setup_logging, get_logger
from src.evaluation.metrics import (
    compute_overall_performance,
    compute_quote_type_accuracy,
    compute_conversation_length_accuracy,
    compute_consecutive_error_runs,
    compute_confidence_drift,
    compute_confidence_baseline
)

setup_logging()
logger = get_logger("EXP016_Analysis")

def load_prediction_dfs(base_dir="results/EXP016"):
    dfs = {}
    for mode in ["teacher_forced", "one_step_autoregressive", "fully_autoregressive"]:
        path = os.path.join(base_dir, mode, "predictions.csv")
        if os.path.exists(path):
            dfs[mode] = pd.read_csv(path)
    return dfs

def load_exp017_prediction_dfs():
    dfs = {}
    exp017_dir = "results/EXP017"
    if not os.path.exists(exp017_dir):
        return dfs
    for subdir in sorted(os.listdir(exp017_dir)):
        path = os.path.join(exp017_dir, subdir, "predictions.csv")
        if os.path.exists(path):
            dfs[subdir] = pd.read_csv(path)
    return dfs

def build_quote_type_map(novels):
    qt_map = {}
    q_info_dir = "data/raw/pdnc/data"
    for novel in novels:
        q_info = pd.read_csv(os.path.join(q_info_dir, novel, "quotation_info.csv"))
        for _, row in q_info.iterrows():
            q_id = row.get("quote_id")
            if not q_id: q_id = f"{novel}_{row.name}"
            q_type = str(row['quoteType'])
            m_phrase = str(row['referringExpression'])
            
            if q_type == 'Implicit':
                final_type = 'Implicit'
            elif q_type == 'Anaphoric':
                final_type = 'Anaphoric'
            elif q_type == 'Explicit':
                if any(c.isupper() for c in m_phrase):
                    final_type = 'Explicit Named'
                else:
                    final_type = 'Explicit Nominal'
            else:
                final_type = 'Implicit'
                
            qt_map[q_id] = final_type
    return qt_map

def load_exp016c_dfs():
    dfs = {}
    path = os.path.join("results/EXP016C/predictions.csv")
    if os.path.exists(path):
        dfs["reverse_one_step_autoregressive"] = pd.read_csv(path)
    return dfs

def load_exp017c_dfs():
    dfs = {}
    path = os.path.join("results/EXP017C/predictions.csv")
    if os.path.exists(path):
        dfs["explicit_anchor_reset"] = pd.read_csv(path)
    return dfs

def main():
    logger.info("Starting EXP016 + EXP017 Analysis...")
    
    # --- EXP016 Analysis ---
    dfs = load_prediction_dfs()
    if not dfs:
        logger.error("No EXP016 predictions found!")
        return
        
    fa_df = dfs.get('fully_autoregressive')
    
    # Analysis 1: Overall Performance
    logger.info("Computing Analysis 1: Overall Performance...")
    perf_df = compute_overall_performance(dfs)
    perf_df.to_csv("results/EXP016/overall_performance.csv", index=False)
    
    # Analysis 2: Quote-Type Accuracy (with N counts)
    logger.info("Computing Analysis 2: Quote-Type Accuracy...")
    novels = dfs['teacher_forced']['novel'].unique()
    qt_map = build_quote_type_map(novels)
    qt_df = compute_quote_type_accuracy(dfs, qt_map, baseline_df=fa_df)
    qt_df.to_csv("results/EXP016/quote_type_accuracy.csv", index=False)
    
    # Analysis 3: Conversation Length
    logger.info("Computing Analysis 3: Conversation-Length Analysis...")
    cl_df = compute_conversation_length_accuracy(dfs)
    cl_df.to_csv("results/EXP016/conversation_length_analysis.csv", index=False)
    
    # Analysis 4: Consecutive Error Runs (corrected terminology)
    logger.info("Computing Analysis 4: Consecutive Error Runs...")
    run_results = []
    for mode, df in dfs.items():
        res = compute_consecutive_error_runs(df)
        res['Mode'] = mode
        run_results.append(res)
    runs_df = pd.DataFrame(run_results)
    runs_df.to_csv("results/EXP016/error_runs_analysis.csv", index=False)
    
    # Analysis 5: Confidence Baseline
    logger.info("Computing Analysis 5: Confidence Baseline...")
    baseline_results = []
    for mode, df in dfs.items():
        res = compute_confidence_baseline(df, mode)
        baseline_results.append(res)
    baseline_df = pd.DataFrame(baseline_results)
    baseline_df.to_csv("results/EXP016/confidence_baseline.csv", index=False)
    
    # Analysis 6: Confidence Drift (with survivorship bias note)
    logger.info("Computing Analysis 6: Confidence Drift & Survivorship...")
    for mode, df in dfs.items():
        drift_df = compute_confidence_drift(df)
        out_path = f"results/EXP016/{mode}/confidence_after_error.csv"
        drift_df.to_csv(out_path, index=False)
        
        # New inclusive and survivorship stats for FA mode
        if mode == "fully_autoregressive":
            from src.evaluation.metrics import compute_confidence_at_error_inclusive, compute_cascade_survivorship
            inc_df = compute_confidence_at_error_inclusive(df)
            inc_df.to_csv("results/EXP016/confidence_at_error_inclusive.csv", index=False)
            
            survivorship = compute_cascade_survivorship(df)
            with open("results/EXP016/cascade_survivorship_audit.json", "w") as f:
                json.dump(survivorship, f, indent=2)
                
    # --- EXP016C Analysis ---
    exp016c_dfs = load_exp016c_dfs()
    if exp016c_dfs:
        logger.info("Computing EXP016C Analysis...")
        os.makedirs("results/EXP016C", exist_ok=True)
        compute_overall_performance(exp016c_dfs).to_csv("results/EXP016C/overall_performance.csv", index=False)
        compute_quote_type_accuracy(exp016c_dfs, qt_map, baseline_df=fa_df).to_csv("results/EXP016C/quote_type_accuracy.csv", index=False)
        
        # Add interaction analysis
        tf_acc = dfs['teacher_forced'][dfs['teacher_forced']['rank'] == 1]['label'].mean()
        fa_acc = dfs['fully_autoregressive'][dfs['fully_autoregressive']['rank'] == 1]['label'].mean()
        os_acc = dfs['one_step_autoregressive'][dfs['one_step_autoregressive']['rank'] == 1]['label'].mean()
        ros_acc = exp016c_dfs['reverse_one_step_autoregressive'][exp016c_dfs['reverse_one_step_autoregressive']['rank'] == 1]['label'].mean()
        
        additive_pred = tf_acc - (tf_acc - os_acc) - (tf_acc - ros_acc)
        interaction_gap = additive_pred - fa_acc
        
        interaction_analysis = {
            "tf_accuracy": f"{tf_acc:.4f}",
            "fa_accuracy": f"{fa_acc:.4f}",
            "os_accuracy": f"{os_acc:.4f}",
            "ros_accuracy": f"{ros_acc:.4f}",
            "additive_prediction": f"{additive_pred:.4f}",
            "interaction_gap_pp": f"{interaction_gap * 100:.2f}"
        }
        with open("results/EXP016C/interaction_analysis.json", "w") as f:
            json.dump(interaction_analysis, f, indent=2)
        
    # --- EXP017 Analysis ---
    exp017_dfs = load_exp017_prediction_dfs()
    if exp017_dfs:
        logger.info("Computing EXP017 Analysis...")
        compute_overall_performance(exp017_dfs).to_csv("results/EXP017/overall_performance.csv", index=False)
        qt_summary_df = compute_quote_type_accuracy(exp017_dfs, qt_map, baseline_df=fa_df)
        qt_summary_df.to_csv("results/EXP017/quote_type_accuracy.csv", index=False)
        
        # Build consolidated quote-type summary Markdown
        md_lines = ["| Mode | Overall | Explicit Named | Anaphoric | Implicit | Δ Implicit vs FA |"]
        md_lines.append("|---|---|---|---|---|---|")
        
        # Include FA and OS from EXP016 for comparison
        exp016_compare = compute_quote_type_accuracy(
            {'fully_autoregressive': dfs['fully_autoregressive'], 'one_step_autoregressive': dfs['one_step_autoregressive']}, 
            qt_map, baseline_df=fa_df
        )
        
        combined_qt = pd.concat([exp016_compare, qt_summary_df])
        
        for _, row in combined_qt.iterrows():
            mode = row['Mode']
            
            # Overall Accuracy
            if mode in dfs:
                ov_acc = float(compute_overall_performance({mode: dfs[mode]}).iloc[0]['Accuracy'])
            elif mode in exp017_dfs:
                ov_acc = float(compute_overall_performance({mode: exp017_dfs[mode]}).iloc[0]['Accuracy'])
            else:
                ov_acc = 0.0
                
            en = float(row.get('Explicit Named', 0)) * 100
            an = float(row.get('Anaphoric', 0)) * 100
            im = float(row.get('Implicit', 0)) * 100
            im_net = row.get('Implicit_Net', 0)
            
            # Δ Implicit vs FA: Since FA is baseline, we can use Implicit_Net / Implicit_N roughly, or just compute % difference
            fa_im = float(exp016_compare.iloc[0].get('Implicit', 0)) * 100 if 'Implicit' in exp016_compare.columns else 0
            im_delta = im - fa_im
            
            md_lines.append(f"| {mode} | {ov_acc * 100:.2f}% | {en:.2f}% | {an:.2f}% | {im:.2f}% | {im_delta:+.2f} pp ({im_net} net) |")
            
        with open("results/EXP017/quote_type_summary.md", "w") as f:
            f.write("\n".join(md_lines) + "\n")
        
        exp017_runs = []
        for mode, df in exp017_dfs.items():
            res = compute_consecutive_error_runs(df)
            res['Mode'] = mode
            exp017_runs.append(res)
        pd.DataFrame(exp017_runs).to_csv("results/EXP017/error_runs_analysis.csv", index=False)
        
        exp017_baselines = []
        for mode, df in exp017_dfs.items():
            exp017_baselines.append(compute_confidence_baseline(df, mode))
        pd.DataFrame(exp017_baselines).to_csv("results/EXP017/confidence_baseline.csv", index=False)
        
    # --- EXP017C Analysis ---
    exp017c_dfs = load_exp017c_dfs()
    if exp017c_dfs:
        from src.evaluation.metrics import compute_anchor_state_drift_stats, compute_post_anchor_window_accuracy
        logger.info("Computing EXP017C Analysis...")
        os.makedirs("results/EXP017C", exist_ok=True)
        compute_overall_performance(exp017c_dfs).to_csv("results/EXP017C/overall_performance.csv", index=False)
        compute_quote_type_accuracy(exp017c_dfs, qt_map, baseline_df=fa_df).to_csv("results/EXP017C/quote_type_accuracy.csv", index=False)
        
        ear_df = exp017c_dfs["explicit_anchor_reset"]
        
        stats = compute_anchor_state_drift_stats(ear_df)
        with open("results/EXP017C/anchor_state_drift.json", "w") as f:
            json.dump(stats, f, indent=2)
            
        window_df = compute_post_anchor_window_accuracy(ear_df, fa_df, qt_map, k=[1,3,5])
        window_df.to_csv("results/EXP017C/post_anchor_implicit_window.csv", index=False)
        
        if not window_df.empty:
            agg_df = window_df.groupby(['window', 'state_drifted']).agg(
                ear_correct=('ear_correct', 'mean'),
                fa_correct=('fa_correct', 'mean'),
                N=('ear_correct', 'count')
            ).reset_index()
            agg_df.to_csv("results/EXP017C/post_anchor_aggregated.csv", index=False)
        
    logger.info("All analyses completed successfully!")

if __name__ == "__main__":
    main()
