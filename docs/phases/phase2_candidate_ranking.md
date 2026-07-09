# Phase 2: Lightweight Candidate Ranking

## Objectives
The objective of Phase 2 is to surpass the symbolic baseline (66.53% Engine Accuracy, 76.26% Oracle Accuracy) by replacing the deterministic `RuleEngine` with a learned scoring model. This model will evaluate the candidate set provided by the frozen Candidate Generator (Window=15) using contextual and discourse features to predict the true speaker.

## Research Questions
1. Can a lightweight learned ranking model exceed the 76.26% symbolic oracle ceiling by resolving deterministic rule conflicts?
2. Which feature families (discourse history, candidate properties, contextual embeddings) provide the highest marginal gain in attribution accuracy?

## Hypotheses

### H1: Learned contextual representations will improve attribution on anaphoric quotations.
- **Motivation:** Deterministic rules crash on anaphoric quotes (49.67% accuracy) because they cannot resolve pronouns (e.g., "he said") to specific candidates.
- **Supporting Evidence:** EXP003 demonstrated that explicit attribution precision is 94.24%, but drops drastically when pronouns are used.
- **Planned Evaluation:** Ablation study comparing the model with and without local context embeddings on the Anaphoric quote subset.
- **Success Criteria:** >15% absolute improvement on Anaphoric quote accuracy compared to the symbolic baseline.

### H2: Learned discourse features will better distinguish dialogue alternation from speaker continuation.
- **Motivation:** EXP003 revealed 31,605 conflicts between the `Dialogue Alternation` (A-B-A-B) and `Previous Speaker` (A-A) heuristics.
- **Supporting Evidence:** These conflicts represent the majority of errors in the `Implicit` quote category.
- **Planned Evaluation:** Evaluate the ranking model's performance on `Implicit` quotes when provided with the previous 3 speaker labels vs no discourse history.
- **Success Criteria:** Reduction in A-A vs A-B-A-B misclassifications by at least 30%.

### H3: Candidate ranking will outperform deterministic rule ordering while remaining lightweight.
- **Motivation:** The frozen `RuleEngine` is rigid and caps out at 66.53%.
- **Supporting Evidence:** The Rule Oracle demonstrates that the rules contain enough information for 76.26% accuracy, but rigid priority ordering loses ~10% of that potential.
- **Planned Evaluation:** Compare a simple logistic regression or lightweight gradient boosting model against the symbolic Engine Accuracy.
- **Success Criteria:** The model exceeds 76.26% overall accuracy while maintaining inference times comparable to the symbolic baseline (<100ms per quote).

## Planned Experiments
- **EXP004 (Feature Extraction):** Establish a robust pipeline for extracting scalar discourse features and lightweight text embeddings for all quotes and candidates.
- **EXP005 (Model Selection):** Train and evaluate Logistic Regression, Random Forest, and LightGBM models.
- **EXP006 (Feature Ablations):** Systematically ablate feature families to isolate their contributions to the primary metric.

## Acceptance Criteria
### Primary Metric
- **Overall Accuracy:** Must exceed the Symbolic Oracle Accuracy of 76.26%.

### Secondary Metrics
- **Quote-Type Accuracy:** Must demonstrate improvement on Anaphoric and Explicit types.
- **Runtime / Memory:** Inference must remain lightweight (suitable for processing a novel in seconds, not hours).

## Risks
- **Data Leakage:** Because characters span the entire novel, train/test splits must be stratified by Novel, not by Quote, to prevent the model from memorizing specific character discourse patterns rather than general rules.
- **Feature Bloat:** Over-engineering features could violate the "Simplicity before complexity" directive.

## Deliverables
1. `src/features/` pipeline for extracting vectors.
2. `src/models/` implementing the Candidate Ranker.
3. Feature Ablation Report.
4. Final Phase 2 Review.
