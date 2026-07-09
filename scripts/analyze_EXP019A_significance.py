import pandas as pd
import numpy as np
import json
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import mcnemar

def main():
    # Load recovery analysis which has both predictions
    df = pd.read_csv("results/EXP019A/recovery_analysis.csv")
    
    # Bootstrap CI for accuracy delta
    n_iterations = 1000
    acc_deltas = []
    imp_deltas = []
    
    # Calculate base performance
    df['exp014_correct'] = (df['baseline_prediction'] == df['gold']).astype(int)
    df['exp019_correct'] = (df['EXP019_prediction'] == df['gold']).astype(int)
    
    implicit_mask = df['quote_type'] != 'Explicit'
    
    overall_acc_delta = df['exp019_correct'].mean() - df['exp014_correct'].mean()
    imp_acc_delta = df.loc[implicit_mask, 'exp019_correct'].mean() - df.loc[implicit_mask, 'exp014_correct'].mean()
    
    for i in range(n_iterations):
        sample = df.sample(frac=1.0, replace=True)
        acc_delta = sample['exp019_correct'].mean() - sample['exp014_correct'].mean()
        acc_deltas.append(acc_delta)
        
        imp_sample = sample[sample['quote_type'] != 'Explicit']
        if len(imp_sample) > 0:
            i_delta = imp_sample['exp019_correct'].mean() - imp_sample['exp014_correct'].mean()
            imp_deltas.append(i_delta)
            
    # McNemar test
    # a: both correct, b: 014 correct 019 wrong, c: 014 wrong 019 correct, d: both wrong
    b = ((df['exp014_correct'] == 1) & (df['exp019_correct'] == 0)).sum()
    c = ((df['exp014_correct'] == 0) & (df['exp019_correct'] == 1)).sum()
    
    table = [[0, b], [c, 0]]
    mc_result = mcnemar(table, exact=False, correction=True)
    
    results = {
        "accuracy_delta": float(overall_acc_delta),
        "accuracy_ci95": [float(np.percentile(acc_deltas, 2.5)), float(np.percentile(acc_deltas, 97.5))],
        "implicit_delta": float(imp_acc_delta),
        "implicit_ci95": [float(np.percentile(imp_deltas, 2.5)), float(np.percentile(imp_deltas, 97.5))],
        "mcnemar": {
            "recovered": int(c),
            "regressed": int(b),
            "p_value": float(mc_result.pvalue)
        }
    }
    
    with open("results/EXP019A/statistical_validation.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Saved statistical validation to results/EXP019A/statistical_validation.json")

if __name__ == "__main__":
    main()
