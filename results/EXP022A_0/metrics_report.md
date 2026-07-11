# EXP022A.0 Relational Speaker GRU Results

## 1. Full Autoregressive Evaluation
**Overall Accuracy**: 72.06% (95% CI: 70.47% - 73.89%)
**Implicit Accuracy**: 67.76% (95% CI: 64.18% - 71.12%)
**Anaphoric Accuracy**: 60.10% (95% CI: 56.21% - 64.07%)
**MRR**: 0.8154 (95% CI: 0.8036 - 0.8272)
**Recall@3**: 88.67%
**LogLoss**: 0.7670

## 2. Memory Ablation (Reset state every quote)
**Overall Accuracy**: 71.42%
**Implicit Accuracy**: 66.58%
**Anaphoric Accuracy**: 59.13%

## 3. Feedback Ablation (Shuffled speaker vectors)
**Overall Accuracy**: 72.02%
**Implicit Accuracy**: 67.50%
**Anaphoric Accuracy**: 60.26%

## 4. Similarity Ablation (Cosine = 0)
**Overall Accuracy**: 72.02%
**Implicit Accuracy**: 67.63%
**Anaphoric Accuracy**: 59.94%

## 5. Similarity Diagnostics (Base Model)
**Mean similarity when predicting correctly**: 0.2176
**Mean similarity of Gold when predicting wrongly**: -0.1330
**Mean similarity of Predicted when predicting wrongly**: 0.0444

## Analysis
- McNemar p-value vs MLP CE Baseline: 4.8518e-06
- MLP CE baseline accuracy: 68.80%
- The Relational GRU achieved a gain of 3.26 pp against the memory-free neural baseline.
- The memory reset ablation caused a drop of 0.64 pp. This isolates the exact contribution of the GRU's recurrence.