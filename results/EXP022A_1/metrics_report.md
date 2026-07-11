# EXP022A.1 Entity-Anchored Relational GRU Results

## 1. Full Autoregressive Evaluation (Normal AR)
**Overall Accuracy**: 78.22% (95% CI: 76.63% - 79.69%)
**Implicit Accuracy**: 82.96% (95% CI: 80.30% - 85.61%)
**Anaphoric Accuracy**: 59.29% (95% CI: 55.34% - 62.96%)
**MRR**: 0.8574 (95% CI: 0.8464 - 0.8686)
**Recall@3**: 91.45%
**LogLoss**: 0.6796

## 2. Memory Ablation (Reset state every quote)
**Overall Accuracy**: 78.34%
**Implicit Accuracy**: 83.36%
**Anaphoric Accuracy**: 59.62%

## 3. Feedback Ablation (Shuffled speaker vectors)
**Overall Accuracy**: 78.26%
**Implicit Accuracy**: 83.09%
**Anaphoric Accuracy**: 59.29%

## 4. Anchor Instability Control
**Overall Accuracy**: 77.82%
**Implicit Accuracy**: 84.01%
**Anaphoric Accuracy**: 57.85%

## 5. Similarity Ablation (Cosine = 0)
**Overall Accuracy**: 78.34%
**Implicit Accuracy**: 83.36%
**Anaphoric Accuracy**: 59.13%

## 6. No-Memory Entity Baseline
**Overall Accuracy**: 77.82%
**Implicit Accuracy**: 82.83%
**Anaphoric Accuracy**: 58.81%

## 7. Similarity Diagnostics (Normal AR)
**Mean similarity when predicting correctly**: 0.0724
**Mean similarity of Gold when predicting wrongly**: 0.2804
**Mean similarity of Predicted when predicting wrongly**: 0.1030
**Similarity Delta (Gold - Pred) when wrong**: 0.1774

## Analysis
- McNemar p-value vs MLP CE Baseline (68.80%): 1.0869e-28
- McNemar p-value vs No-Memory Entity Baseline: 4.7950e-01

- **Gain vs MLP CE state-free**: 9.42 pp
- **Gain vs No-Memory Entity Baseline (Memory Effect)**: 0.40 pp
- **Memory Reset Ablation drop**: -0.12 pp
- **Shuffled Feedback Ablation drop**: -0.04 pp
- **Anchor Instability Control drop**: 0.40 pp

### Success Criteria Checks
- Normal - Reset >= 1.5 pp: FAIL (-0.12 pp)
- Normal - Shuffled Feedback >= 1.0 pp: FAIL (-0.04 pp)
- Normal - No-Memory Entity Baseline >= 1.5 pp: FAIL (0.40 pp)