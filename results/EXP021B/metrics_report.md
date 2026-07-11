# EXP021B Speaker-Feedback GRU Results

## 1. Full Autoregressive Evaluation
**Overall Accuracy**: 66.06% (95% CI: 64.27% - 67.93%)
**Implicit Accuracy**: 52.16% (95% CI: 48.53% - 55.46%)
**Anaphoric Accuracy**: 51.12% (95% CI: 47.32% - 54.70%)
**MRR**: 0.7797 (95% CI: 0.7677 - 0.7920)
**Recall@3**: 87.96%
**LogLoss**: 0.9132

## 2. Memory Ablation (Reset state every quote)
**Overall Accuracy**: 66.14%
**Implicit Accuracy**: 52.42%
**Anaphoric Accuracy**: 51.12%

## 3. Feedback Ablation (Speaker vectors = Zeros)
**Overall Accuracy**: 66.18%
**Implicit Accuracy**: 52.69%
**Anaphoric Accuracy**: 50.96%

## 4. Feedback Ablation (Shuffled speaker vectors)
**Overall Accuracy**: 66.06%
**Implicit Accuracy**: 52.16%
**Anaphoric Accuracy**: 51.12%

## Analysis
- McNemar p-value vs MLP CE Baseline: 3.8315e-06
- MLP CE baseline accuracy: 69.20%
- The GRU achieved a loss of 3.14 pp against the memory-free neural baseline.
- The memory reset ablation caused a rise of 0.08 pp. This isolates the exact contribution of the GRU's recurrence.