# Current Project Status

## Current Phase: 
Phase 3 — Semantic Context Representations
**Current Step:** Stage 3A (EXP010: Semantic Error Taxonomy)
**Active Hypothesis (H10):** Residual errors cluster into a small number of recurring semantic phenomena rather than forming an unstructured set.

## Recently Completed
- **EXP004B (Linear Evaluation):** Logistic Regression achieved 82.81% accuracy, proving strong linearly separable signal.
- **EXP004C (Feature Ablation):** Candidate features dominate predictive signal; Symbolic features degraded linear performance.
- **EXP004D (Symbolic Investigation):** Confirmed symbolic rules introduce multicollinearity and noisy deterministic boundaries.
- **EXP005 (Representation Analysis):** Established that 3 continuous candidate features (`is_recent_mention`, `is_previous_speaker`, `is_explicit_mention`) recover 84.10% accuracy, while the remaining 10 features add only 0.13%.

## Immediate Next Step
- **Action:** Create `scripts/run_exp010_semantic_error_taxonomy.py`.
- **Goal:** Extract a stratified sample of 200 residual errors (Both Fail, GBM Only, Explicit Present + Fail) for manual open coding to determine what semantic phenomena are missing from the explicit representation.

## Long-term Roadmap
- EXP006: Contextual Redundancy Analysis
- EXP007: Per-Novel Generalization
- Decision Gate
- Phase 2D: Lightweight Neural Models (Conditional)
