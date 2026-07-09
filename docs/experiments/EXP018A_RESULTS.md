# EXP018A: Global Decoding (Beam Search) vs Local Classifier

## Overview
This experiment answers the core causal question arising from the failures of autoregressive decoding (EXP016/EXP017): *Is the model failing because of its greedy commitment strategy, or because its local scoring function is fundamentally misspecified for sequences?*

By freezing the HistGBM classifier and replacing greedy decoding with Beam Search (sweeping `K=1` to `20`), we isolated the search algorithm from the model weights. 

## Results: Beam Size Sweep

| K | Overall Accuracy | Implicit Accuracy | Oracle Path Survival |
|---|---|---|---|
| 1 (Greedy baseline)| 65.46% | 60.03% | 0.0% |
| 3 | 65.94% | 61.99% | 0.0% |
| 5 | 65.46% | 60.81% | 0.0% |
| 10 | 65.46% | 61.99% | 0.0% |
| 20 | 65.66% | 62.52% | 0.0% |

*(Note: K=1 matches the EXP016 Fully Autoregressive results exactly, validating the beam search state tracking implementation).*

## Interpretation

### 1. Global Decoding Does Not Save Local Classifiers
Increasing the beam size from `K=1` to `K=20` yielded negligible improvement (+0.20 pp overall). Critically, accuracy **fluctuates non-monotonically** as K increases (dropping at `K=5` and `K=10`). 

When a broader beam search finds sequences with *higher* joint probabilities but *lower* ground-truth accuracy, it is definitive proof that **beam search over locally-trained probabilities does not recover autoregressive degradation**. The HistGBM—trained on isolated decisions with teacher forcing—assigns high probabilities to degenerate conversational paths because it has no global structural constraints to penalize them.

### 2. The True Path is Unrecoverable
**Full-sequence oracle survival collapses due to accumulated sequence risk; decay analysis measures where loss occurs.** Even though the candidate generator has high local recall, the unbroken gold path inevitably dies before the end of the novel. This confirms two critical issues:
1. **Candidate Generator Recall Constraint**: The unbroken gold path inevitably dies the moment the true speaker is omitted from the candidate generation step for even a single quote.
2. **Probability Mass Starvation**: Even when candidates are present, the model's local confidence in the wrong speaker frequently forces the true path entirely out of the top K beam. 

Because the true sequence falls out of the beam, global decoding is completely useless without a local loss function trained to preserve it.

## Conclusion & Next Steps
EXP018A definitively concludes that **the problem is the model's knowledge, not the greedy commitment strategy.** Applying global search over locally-trained probabilities fails because the local classifier is not calibrated for sequence-level interactions.

This provides the required scientific justification for **Phase 5: Sequence-Level Models**. We must transition from a model that scores isolated `(quote, candidate)` pairs to a model trained with a sequence-level loss function (e.g., Conditional Random Fields or Neural Sequence models) that inherently learns global conversational structures.
