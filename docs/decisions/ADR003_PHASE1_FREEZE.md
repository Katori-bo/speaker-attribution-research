# ADR 003: Phase 1 Component Freeze

**Date:** 2026-07-01
**Status:** Accepted

## Context
Phase 1 (Symbolic Baseline) of the speaker attribution project has concluded. Over the course of four experiments (EXP001, EXP002, EXP002b, EXP003), we systematically established, tested, and validated the deterministic infrastructure required to evaluate speaker attribution models on the PDNC dataset. To maintain rigorous scientific control moving into Phase 2, we must freeze these foundational components.

## Decision
The following components are officially frozen and must not be altered without a new, explicit architectural experiment:

### 1. Dataset & Ingestion Pipeline
- **Status:** Frozen
- **Reason:** Validated by EXP001. We confirmed PDNC contains exactly 37,131 quotes, 0 unknown speakers, and structural integrity is intact.
- **Configuration:** Fingerprint `4a0549a145ea8500bea395a0b7d947e3589d28ff346bc9022380f34530387ad4`.

### 2. Candidate Generator
- **Status:** Frozen
- **Reason:** Validated by EXP002b. Expanding the discourse window proved necessary and sufficient for establishing a theoretical baseline ceiling of 92.76%.
- **Configuration:** Window = 15 previous speakers. Alias normalization disabled (negligible impact).

### 3. Symbolic Rule Engine (Evaluator & Priority Engine)
- **Status:** Frozen
- **Reason:** Fully evaluated by EXP003. The decoupled architecture successfully isolates rule performance from priority ordering, proving a Rule Oracle Accuracy of 76.26% and Engine Accuracy of 66.53%.
- **Configuration:** Priority Order: Explicit -> Dialogue Alternation -> Previous Speaker -> Nearest Mention.

### 4. Experiment Recorder & Evaluation Pipeline
- **Status:** Frozen
- **Reason:** Validated across all Phase 1 experiments. The fail-fast mechanisms, Git commit snapshotting, and JSON metric outputs produce fully reproducible artifacts.

## Consequences
- Any performance improvements in Phase 2 must come from machine-learned models, not from tweaking these frozen deterministic heuristics.
- The Phase 1 Engine Final Accuracy (66.53%) is the official baseline to beat.
- The Candidate Generator Ceiling (92.76%) is the theoretical maximum for any downstream ranking model until Candidate Generation itself is revisited.
