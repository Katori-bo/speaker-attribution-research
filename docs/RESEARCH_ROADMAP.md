# Research Roadmap

## Purpose

This document describes the experimental decision tree for the project.

It is not an implementation plan. It describes how the outcome of each phase determines the direction of future work.

The roadmap should be read after understanding the research questions (02_RESEARCH_QUESTIONS.md) and the hypotheses (07_RESEARCH_HYPOTHESES.md).

---

## Phase 1 — Heuristic Baseline

**Current Question:**
How much speaker attribution accuracy is achievable without machine learning?

**Possible Outcomes:**

* **High accuracy on explicit quotes (>70%):** Confirms that rule-based methods handle simple cases well. The remaining errors define the challenge for learned models.
* **Low accuracy on explicit quotes (<50%):** Suggests that even explicit attribution patterns are more complex than expected in literary text. May require revisiting dialogue detection.
* **Very low accuracy on implicit quotes:** Expected. Defines the gap that contextual models must close.

**Next Experiment:**
Phase 2 — External Baseline (BookNLP).

**Alternative Path:**
If dialogue detection itself fails significantly, a sub-experiment on dialogue extraction quality may be required before proceeding.

**Decision Criteria:**
Proceed to Phase 2 regardless of outcome. Phase 1 results become the floor for all future comparisons.

---

## Phase 2A — Feature Representation

**Current Question:**
How can we extract contextual information into a fixed schema suitable for candidate ranking without leaking future information?

**Deliverable:**
A reproducible dataset of `(quote, candidate)` pairs with extracted features categorized into Lexical, Candidate, Discourse, Conversation, and Symbolic families.

**Next Experiment:**
EXP004A — Feature Audit.

**Decision Criteria:**
Proceed to EXP004A once dataset generation is deterministic and passes leakage tests.

---

## EXP004A — Feature Audit

**Current Question:**
Are the extracted features numerically stable, well-distributed, and free of trivial bugs?

**Deliverable:**
A comprehensive `feature_summary.csv` analyzing variance, missingness, and label correlation for every feature.

**Next Action:**
Freeze Feature Schema (ADR004).

**Decision Criteria:**
If bugs or malformed features are found, fix them and regenerate. Once the audit is clean, freeze the schema and proceed to EXP004B.

---

## EXP004B — Logistic Regression Evaluation

**Current Question:**
Does the representation itself contain predictive signal beyond symbolic heuristics when evaluated with a simple linear model?

**Possible Outcomes:**
* **Significant improvement over heuristics:** Validates the representation.
* **No improvement:** Suggests the representation is insufficient, or non-linear interactions are strictly required.

**Next Experiment:**
EXP004C — Feature Family Ablations.

**Decision Criteria:**
Proceed to EXP004C regardless of accuracy, as the ablation will explain the model's behavior.

---

## EXP004C — Feature Family Ablations

**Current Question:**
Which contextual feature families (Lexical, Candidate, Discourse, Conversation) contribute the most to the model's accuracy?

**Deliverable:**
An ablation study quantifying the performance drop when each feature family is removed.

**Next Experiment:**
Phase 2D — Lightweight Neural Models (Optional).

**Decision Criteria:**
If the linear model performs well and ablations are interpretable, proceed to test if Neural Models (Phase 2D) can capture non-linear interactions. If the linear model fails completely, conduct deep error analysis.

## EXP004D — Symbolic Feature Investigation

**Current Question:**
Why did ablating the Symbolic feature family improve linear baseline accuracy? 

**Deliverable:**
An investigation evaluating competing hypotheses: (H1) Multicollinearity, (H2) Redundant information, (H3) Noisy rules, (H4) Excessively coarse binary encoding. Creates a direct bridge by comparing symbolic rules against continuous candidate features.

**Next Experiment:**
EXP005 — Representation Analysis.

**Decision Criteria:**
Proceed to EXP005 once the degradation is understood.

---

## EXP005 — Representation Analysis

**Current Question:**
Which individual contextual signals are indispensable, which are redundant, and how well-calibrated is the representation?

**Deliverable:**
- Coefficient Analysis (Coefficients, Standard Errors, 95% CIs, p-values).
- Single-Feature Ablations.
- Representation Sufficiency Curve (forward feature selection).
- Calibration Metrics (ROC-AUC, PR-AUC, Brier Score, ECE, Reliability Diagrams).

**Next Experiment:**
EXP006 — Feature Redundancy & Interaction.

**Decision Criteria:**
Proceed to EXP006 once the individual contribution of all features is statistically quantified.

---

## EXP006 — Contextual Redundancy Analysis

**Current Question:**
Are the remaining contextual features redundant, or do they contain complementary information masked by the linear model?

**Deliverable:**
Four targeted sub-experiments:
- **A. Correlation Structure:** Phi, Point-biserial, and Pearson matrices.
- **B. Conditional Importance:** $\Delta$ Accuracy, $\Delta$ ROC-AUC, $\Delta$ ECE, and $\Delta$ Log Loss when adding individual features to the Top 3.
- **C. Residual Error Taxonomy:** Categorization of residual failure modes.
- **D. Interaction Test:** Hypothesis-driven tests (e.g., Previous Speaker × Context Length).
- **E. Stability Analysis:** Forward and Backward cross-validated feature selection.

**Next Experiment:**
EXP007 — Per-Novel Generalization.

**Decision Criteria:**
Proceed to EXP007 once redundancy and failure modes are mapped.

---

## EXP007 — Per-Novel Generalization

**Current Question:**
Does the "3-feature sufficiency" hypothesis hold across individual novels, or do novels with larger casts/complexity require richer representations?

**Deliverable:**
Novel-level $\Delta$ Accuracy (Top 3 vs Full) plotted against cast size and dialogue complexity.

**Next Experiment:**
Decision Gate.

**Decision Criteria:**
If EXP006 uncovers useful non-linear interactions OR EXP007 shows the minimal representation failing on complex novels, proceed to EXP009. Otherwise, proceed to EXP008 (Nonlinear Sanity Check).

---

## EXP008 — Nonlinear Sanity Check

**Current Question:**
Does a lightweight nonlinear learner (HistGradientBoosting) extract meaningful additional predictive signal from the current explicit representation compared to a linear model?

**Deliverable:**
Comparison of Accuracy, ROC-AUC, PR-AUC, Log Loss, and ECE between Logistic Regression and HistGradientBoosting (untuned). Permutation importance analysis if the nonlinear model succeeds.

**Next Experiment:**
EXP009 — Understanding Nonlinear Gain.

**Decision Criteria:**
EXP008 triggered Outcome B: HistGBM provided a massive gain in PR-AUC. Proceed to EXP009 to dissect this gain before moving to Phase 4.

---

## EXP009 — Understanding Nonlinear Gain

**Current Question:**
Why does HistGradientBoosting improve ranking quality (PR-AUC) significantly while providing only a modest gain in top-1 accuracy?

**Deliverable:**
Four targeted analyses:
- **A. Prediction Comparison:** Taxonomy of `GBM Only` successes vs `LR Only` successes.
- **B. Ranking Metrics:** MRR, Mean Rank, Recall@1, Recall@3.
- **C. Residual Taxonomy:** Did the nonlinear model solve pronoun/scene errors, or just calibrate better?
- **D. Interaction Interpretation:** 2D Partial Dependence Plots for `previous_speaker` × `context_length`.

**Next Experiment:**
Stage 3A (Semantic Error Taxonomy).

**Decision Criteria:**
Phase 2 is considered complete based on the evidence from EXP006–EXP009. The next research direction is Phase 3: Semantic Context Representations.

---

## Phase 3 — Semantic Context Representations

**Current Question:**
Which semantic phenomena account for the residual errors left by explicit contextual representations, and what is the minimal semantic representation required to recover them?

### Stage 3A: EXP010 Semantic Error Taxonomy
- **H10:** Residual errors cluster into a small number of recurring semantic phenomena rather than forming an unstructured set.
- **Methodology:** Stratified sampling (100 Both Fail, 50 GBM Only, 50 Explicit Present + Fail) and open coding.
- **Acceptance:** A stable taxonomy where every residual error is assigned exactly one primary semantic category.

### Stage 3B: Representation Design (Decision Gate)
- **Constraint:** No semantic representation may be implemented until the design has been justified using the EXP010 taxonomy.
- **Methodology:** Map taxonomy phenomena to Representation Families (e.g., Sentence, Contextual Token, Paragraph) paired with heuristic baselines.

### Stage 3C: EXP011 Semantic Audit & EXP012 Frozen Features
- **H11:** Semantic representations contain complementary information and do not duplicate explicit features.
- **H12:** Different representation families preferentially recover different semantic phenomena.
- **Acceptance:** Deterministic extraction, zero leakage. Reporting recovery rate per phenomenon and computational cost.

### Stage 3D: EXP013 Semantic Sufficiency
- **H13:** Only a small amount of semantic information is required before performance reaches a new plateau.
- **Methodology:** Explicit + Semantic Sufficiency Curve, followed by an "Error Overlap" final analysis.

---

## Phase 5 — Final Evaluation

**Current Question:**
How does the final pipeline perform on a held-out literary dataset?

**Deliverable:**
Complete scientific evaluation answering all research questions.

This phase produces conclusions, not new experiments.

---

## Roadmap Summary

```
 Heuristic Baseline
         ↓
 Feature Representation (Phase 2A)
         ↓
 Feature Audit (EXP004A)
         ↓
 Logistic Regression (EXP004B)
         ↓
 Feature Ablations (EXP004C)
         ↓
 Symbolic Investigation (EXP004D)
         ↓
 Representation Analysis (EXP005)
         ↓
 Contextual Redundancy (EXP006)
         ↓
 Per-Novel Generalization (EXP007)
         ↓
 Decision Gate
         ↓
 Nonlinear Sanity Check (EXP008)
         ↓
 Understanding Nonlinear Gain (EXP009)
         ↓
 Semantic Error Taxonomy (EXP010)
         ↓
 Semantic Audit & Evaluation (EXP011 & EXP012)
         ↓
 Semantic Sufficiency Curve (EXP013)
         ↓
 Final Evaluation (Phase 5)
 ```

Every transition between phases depends on the outcome of the previous phase.

No phase should be skipped without documented justification.
