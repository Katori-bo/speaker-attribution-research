# EXP009: Understanding Nonlinear Gain

## Part A: Prediction Comparison

| Category     |   Count |
|:-------------|--------:|
| Both Correct |    1900 |
| Both Wrong   |     291 |
| GBM Only     |      78 |
| LR Only      |      52 |

**Characteristics of GBM-Only Successes:**
- Avg Conv Length (GBM Only): 634.72
- Avg Conv Length (Both Correct): 464.66
- Avg Context Length (GBM Only): 15.46
- Avg Context Length (Both Correct): 12.57

## Part B: Ranking Quality

| Model            |      MRR |   Mean_Rank |   Recall@1 |   Recall@3 |
|:-----------------|---------:|------------:|-----------:|-----------:|
| Logistic (Top 3) | 0.897755 |     1.37398 |   0.841448 |   0.947436 |
| HistGBM (All 13) | 0.907136 |     1.33563 |   0.852219 |   0.953037 |

## Part C: Residual Error Taxonomy (HistGBM)

Did GBM solve the semantic errors? (Hint: No, if 'Gold has no explicit signals' is still the largest class).

| error_type                            |   count |
|:--------------------------------------|--------:|
| Gold has no explicit signals          |     232 |
| Confused Mention for Previous Speaker |      67 |
| Unknown                               |      29 |
| Long Context Narration                |      15 |

## Part D: Targeted Interaction Interpretation

![PDP Interaction](/home/Aditya/speaker-attribution-research/results/EXP009/pdp_interaction.png)
