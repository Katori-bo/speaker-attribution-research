# Evaluation Metric Audit: EXP008 vs EXP012

## The Discrepancy
- **EXP008 Reported Accuracy:** ~85.2%
- **EXP012 Reported Accuracy:** ~80.7%
- **Reviewer Question:** Where did the 4.5% go? Was there feature pollution or an unfair split change?

## Investigation Findings
An audit of `run_exp008_nonlinear_sanity_check.py` and `run_exp012.py` reveals that the discrepancy was **not** caused by feature pollution, differing random splits, or different test sets. Both experiments used the exact same test set (a deterministic novel-level split with 3 novels).

The difference is entirely due to **Aggregation Methodology**:
1. **EXP008 (Conditional Accuracy):** The evaluation loop silently dropped quotes where the gold speaker was completely missing from the candidate set. It calculated accuracy as: `Correct Top-1 / Quotes with Gold Candidate`.
   - Total quotes in test: 2,516
   - Quotes containing gold candidate (Oracle): 2,321
   - EXP008 Correct Top-1: 1,982
   - EXP008 Reported Accuracy: 1982 / 2321 = **85.39%**

2. **EXP012 (Unconditional Accuracy):** The `get_ranking_metrics` function strictly evaluates over **all** quotes. Quotes missing the gold candidate are correctly penalized as failures (since the model cannot predict a candidate it wasn't given).
   - If EXP008 was evaluated unconditionally, its accuracy would be: 1982 / 2516 = **78.78%**
   - EXP012 added coreference features, raising this unconditional accuracy to **80.72%**.

## Conclusion
The model did not degrade. In fact, EXP012 significantly outperformed EXP008 (+1.94% absolute gain). The "drop" was merely an artifact of moving from an Oracle-conditional metric to a strict, rigorous, unconditional accuracy metric.
