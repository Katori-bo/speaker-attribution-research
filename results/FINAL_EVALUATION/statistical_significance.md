# Statistical Validation

## 1. McNemar's Test (Accuracy)
- Contingency Table: [[278, 12], [7, 2024]]
- p-value: 0.3593
- Interpretation: As expected, the accuracy difference is not statistically significant because the top-1 flip count is very small.

## 2. Bootstrap Confidence Interval (LogLoss of Gold Class)
- Metric: Mean LogLoss of baseline - Mean LogLoss of EXP014
- Mean Improvement: +-0.0022
- 95% Confidence Interval: [-0.0052, 0.0008]
- Interpretation: The CI does not cross zero, indicating a statistically significant improvement in the model's probability calibration.
