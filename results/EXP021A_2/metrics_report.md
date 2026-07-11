# EXP021A.2 Ranking MLP (CrossEntropy) Results

**Overall Accuracy**: 68.80% (95% CI: 66.97% - 70.75%)
**Implicit Accuracy**: 57.93% (95% CI: 54.50% - 61.34%)
**Anaphoric Accuracy**: 58.97% (95% CI: 55.15% - 62.87%)
**MRR**: 0.8006 (95% CI: 0.7895 - 0.8130)
**Recall@3**: 89.94%
**LogLoss (CrossEntropy)**: 0.7907

## Statistical Significance (vs HistGBM AR)
- HistGBM AR Accuracy: 65.46%
- McNemar p-value: 2.6552e-04
- **Conclusion**: The performance difference is statistically significant.

## Analysis
This establishes the `MLP CE (state-free)` baseline. Any gains made by the GRU will be measured directly against these numbers to isolate the contribution of recurrent memory.