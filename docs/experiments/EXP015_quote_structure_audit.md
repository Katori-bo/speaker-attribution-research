# EXP015 — Quote Structure & Saturation Audit

## Goal
Verify whether the EXP014 plateau (~80.9%) represents true representation saturation or whether implicit dialogue is hiding an unsolved subproblem.

This is a **diagnostic experiment only**.
No new features.
No architecture changes.
No model tuning.

## Baseline
Baseline = EXP014

### Features:
- ✓ Top-3 discourse
- ✓ HistGBM
- ✓ Coreference
- ✓ Alias mapping
- ✓ Named syntactic attribution
- ✗ Addressee state (Rejected)

## Research Question
Does lightweight contextual representation saturate across all quote types, or only explicit attribution cases?

## Findings
We mapped all 2,516 test set quotes directly to their **gold-standard quote type annotations** provided in the PDNC `quotation_info.csv` files (`Explicit`, `Anaphoric` [Pronoun], and `Implicit`). 

The breakdown of performance is as follows:

| System  | Explicit Named | Explicit Nominal | Explicit Pronoun | Implicit |
| ------- | -------------- | ---------------- | ---------------- | -------- |
| BookNLP | 0.9285         | 0.8000           | 0.7740           | 0.5216   |
| Top3    | 0.8266         | 0.3000           | 0.5833           | 0.8532   |
| EXP012  | 0.8391         | 0.3000           | 0.6587           | 0.8886   |
| EXP014  | 0.8418         | 0.3000           | 0.6603           | 0.8899   |

**Conclusion:** 
The results hold: **Implicit accuracy is genuinely 88.99%**.

As hypothesized, long alternating conversations without attribution cues (Implicit) are heavily patterned. Features like `candidate_is_previous_speaker`, `recent_mention_count`, and `chain_recency` are **excellent** at predicting alternation. The Top-3 baseline alone achieves 85% on these quotes.

Meanwhile, **Explicit Pronouns** (e.g., `"Yes," she said.`) represent the real bottleneck at **66.03%**. In dense scenes with multiple characters of the same gender, simple recency and coreference fail to resolve the ambiguity, whereas implicit back-and-forth chains remain deterministic.

This firmly establishes **Case 2**: The representation is saturated for implicit quotes. The remaining errors are tied to semantic ambiguity in explicit pronouns, which structural heuristics cannot overcome.
