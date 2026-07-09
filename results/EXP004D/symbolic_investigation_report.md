# EXP004D: Symbolic Feature Investigation Report

## Hypothesis 1 & 2: Multicollinearity and Redundancy
Do symbolic features perfectly duplicate continuous candidate features?

| Symbolic Feature | Candidate Feature | Pearson Correlation |
|------------------|-------------------|---------------------|
| symbolic_alternation_rule_fired | candidate_is_previous_speaker | 0.9026 |
| symbolic_explicit_rule_fired | candidate_is_explicit_mention | 0.9959 |

## Hypothesis 3: Noisy Rules
Are the symbolic rules introducing noise (i.e. firing confidently on incorrect candidates)?

| Symbolic Feature | Fired Count | Precision (Correct when fired) |
|------------------|-------------|--------------------------------|
| symbolic_alternation_rule_fired | 29434 | 68.75% |
| symbolic_explicit_rule_fired | 2581 | 93.92% |

## Conclusion
The data strongly supports **H1 (Multicollinearity)** and **H3 (Noisy Rules)**.
1. **Multicollinearity (H1):** The symbolic features are almost perfectly collinear with the explicit candidate features (e.g., `symbolic_explicit_rule_fired` and `candidate_is_explicit_mention` have a 0.9959 correlation). This collinearity confuses the linear model's weight assignment.
2. **Noisy Rules (H3):** The `symbolic_alternation_rule_fired` feature fires nearly 30,000 times, but is only correct 68.75% of the time. The deterministic rule lacks the nuance of the linear model, forcing a noisy binary signal into the representation. 

By removing the symbolic features, we eliminate multicollinearity and noisy heuristics, allowing the model to smoothly weight the underlying continuous variables.
