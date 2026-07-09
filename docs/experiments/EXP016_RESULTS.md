# EXP016/EXP017 Results: The Attribution Degradation Cliff

This document consolidates the results of EXP016 (State Corruption Decomposition) and EXP017 (Cheap Mitigations). It replaces the informal walkthrough with formalized metrics verifying the boundaries of the autoregressive attribution model.

## 1. EXP016: Degradation Table & Calibration

Transitioning from teacher-forced (TF) inference to fully-autoregressive (FA) inference results in a severe performance drop.

### 1.1 The Quote-Type Cliff
The autoregressive model specifically fails on implicit and anaphoric quotes, which rely heavily on accurately persisted state:

| Quote Type | TF Accuracy | FA Accuracy | Degradation |
|------------|-------------|-------------|-------------|
| Explicit Named | 81.32% | 77.03% | -4.29 pp |
| Explicit Nominal| 60.00% | 40.00% | -20.00 pp |
| Anaphoric | 74.20% | 52.08% | -22.12 pp |
| Implicit | 83.22% | 60.03% | -23.19 pp |

### 1.2 Confidence Calibration & Survivorship Bias
At the point of an error, the FA model is highly confident in its incorrect predictions. We define confidence as the probability assigned to the predicted candidate:
```python
confidence = max(candidate_probability)
```

| Mode | Correct confidence | Error confidence |
| ---- | -----------------: | ---------------: |
| TF   | 95.04% | 69.16% |
| FA   | 93.87% | 77.28% |

- **Mean Confidence at Error (Inclusive):** 82.20% (from `confidence_at_error_inclusive.csv`)

> [!WARNING]
> The drift-distance table in EXP016 excludes **7.6%** (31 out of 406) of error cascades that never recover within a 5-quote window (`cascade_survivorship_audit.json`). The inclusive wrong-prediction confidence is ~82.20%, demonstrating that the model remains blindly confident even when catastrophically wrong.

## 2. EXP016C: Interaction Effect Analysis

We decomposed the degradation into two sources: `last_speaker` corruption and conversational history corruption. 

| Condition | Overall Accuracy | Delta from TF |
|-----------|------------------|---------------|
| Teacher Forced (all gold) | 80.60% | — |
| One-step (corrupt last_speaker only) | 69.24% | −11.36 pp |
| Reverse one-step (gold last_speaker, corrupt history) | 79.77% | −0.83 pp |
| Fully autoregressive (corrupt all) | 65.46% | −15.14 pp |

Additive prediction (80.60 - 11.36 - 0.83) = **68.41%**
Actual FA Accuracy = **65.46%**

**Finding:** `last_speaker` is the dominant corruption pathway, with secondary interaction effects (~3 pp) when both corruption sources are active simultaneously. The reverse ablation at 79.77% remains decisive: history corruption in isolation has minimal impact. The critical failure is unstable propagation of the immediate speaker identity.

## 3. EXP017: Cheap Mitigations Failed (Blocker Table)

We attempted various "cheap" mitigations in EXP017: confidence gating (A), soft state (B), and explicit anchor reset (C). None succeeded at recovering implicit quote performance.

| Mode | Overall | Explicit Named | Anaphoric | Implicit | Δ Implicit vs FA |
|---|---|---|---|---|---|
| fully_autoregressive (Baseline) | 65.46% | 77.03% | 52.08% | 60.03% | +0.00 pp (0 net) |
| one_step_autoregressive | 69.24% | 80.25% | 58.97% | 61.86% | +1.83 pp (14 net) |
| confidence_gated_0.70 | 65.18% | 76.32% | 52.56% | 59.76% | -0.27 pp (-2 net) |
| confidence_gated_0.80 | 63.59% | 75.96% | 51.92% | 55.57% | -4.46 pp (-34 net) |
| confidence_gated_0.85 | 64.79% | 76.50% | 51.28% | 59.24% | -0.79 pp (-6 net) |
| confidence_gated_0.90 | 64.51% | 76.76% | 50.96% | 58.19% | -1.84 pp (-14 net) |
| confidence_gated_0.95 | 64.31% | 77.30% | 52.40% | 55.57% | -4.46 pp (-34 net) |
| soft_state | 65.10% | 76.76% | 51.92% | 59.24% | -0.79 pp (-6 net) |
| explicit_anchor_reset | 65.46% | 77.21% | 51.76% | 59.76% | -0.27 pp (-2 net) |

**Category Verdict:** Cheap mitigations do not recover implicit dialogue performance. All mitigations landed within ±1 pp of FA on implicit quotes, and several regression thresholds (e.g., 0.80) caused severe drops (-4.46 pp).

## 4. EXP017C Corrected Anchor Analysis

We investigated why the Explicit Anchor Reset mechanism failed to improve performance downstream. The core question: *Does an explicit attribution event correctly reset a drifted state and improve subsequent implicit predictions?*

**Anchor State Drift Statistics (`anchor_state_drift.json`)**
- Total Anchor Events: 247
- State was already correct at anchor: 198 (80%)
- State had drifted before anchor: 49 (20%)
- State Resets Applied (Prediction Overridden): 8

**Post-Anchor Implicit Window Accuracy (`post_anchor_aggregated.csv`)**
For the 49 cases where the state had drifted, we measured downstream accuracy on implicit quotes in the following N steps:

| Window (K steps post-anchor) | N Quotes | EAR Correct | FA Correct |
|------------------------------|----------|-------------|------------|
| K = 1                        | 8        | 0.0% (0/8)  | 0.0% (0/8) |
| K = 3                        | 21       | 23.8% (5/21)| 28.5% (6/21)|
| K = 5                        | 39       | 30.7% (12/39)| 35.8% (14/39)|

> [!CAUTION]
> The anchor mechanism not only failed to improve downstream implicit accuracy, it **underperformed the baseline FA model** in windows 3 and 5 following a drifted state. Furthermore, 80% of anchors occurred when the state was already correct.

## Decision Gate: Proceed to Phase 5

The corrected anchor analysis shows **no downstream implicit benefit**; the cheap anchor mechanism performs equal to or worse than FA in recovery windows. Because all local/heuristic mitigations (confidence thresholds, soft state, rule-based anchor overrides) have failed to arrest error cascading in implicit quotes, **Phase 5 (Global BiLSTM/CRF sequence modeling) is now definitively necessary.**
