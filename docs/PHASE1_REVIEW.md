# Phase 1 Summary Report: Symbolic Baseline

## 1. Objectives
The objective of Phase 1 was to establish a rigorous, reproducible, and deterministic baseline for speaker attribution on the Project Dialogism Novel Corpus (PDNC). Instead of immediately deploying machine learning models, we sought to quantify the exact limits of symbolic reasoning to identify precisely where statistical models are required.

## 2. Experiments Conducted
- **EXP001 (Dataset Characterization):** Validated the integrity of the PDNC dataset, identifying 37,131 quotes across 28 novels with a heavy long-tail speaker distribution and 0 unlabeled quotes.
- **EXP002 (Candidate Generation):** Evaluated baseline candidate generation (Window=5), revealing an 88.29% recall ceiling, proving that missing candidates are a primary bottleneck.
- **EXP002b (Candidate Ablations):** Isolated variables to test Window=15, Window=30, and Alias Normalization. Proved that Window=15 is optimal (92.76% recall) and alias normalization has negligible impact.
- **EXP003 (Symbolic Rule Evaluation):** Decoupled rule evaluation from prediction to measure the independent coverage, precision, and contribution of deterministic attribution rules (Explicit, Dialogue Alternation, Previous Speaker, Nearest Mention).

## 3. Major Findings
| Finding | Supporting Experiment | Evidence |
|---------|-----------------------|----------|
| Candidate generation is the first bottleneck | EXP002 | 88.29% ceiling with Window=5 |
| Window 15 provides the best recall/efficiency trade-off | EXP002b | 92.76% recall |
| Alias normalization has negligible impact on baseline | EXP002b | +0.03% recall |
| Symbolic heuristics are insufficient alone | EXP003 | 76.26% Rule Oracle Accuracy |
| Dialogue alternation (A-B-A-B) dominates symbolic reasoning | EXP003 | 85.12% coverage, 69.20% of wins |
| Anaphoric quotes remain fundamentally difficult | EXP003 | 49.67% accuracy for pronouns |
| Nearest internal mention is anti-correlated with speaker | EXP003 | 0.58% precision |

## 4. Architectural Decisions
- Implemented isolated, immutable `EXPxxx` run directories for exact reproducibility.
- Decoupled `RuleEvaluator` (independent analysis) from `RuleEngine` (priority decision-making) to prevent obfuscating rule performance.
- Instituted a formal Architecture Decision Record (ADR) process for freezing components.

## 5. Frozen Components
*(See `docs/decisions/ADR003_PHASE1_FREEZE.md`)*
- Dataset & Ingestion Pipeline
- Candidate Generator (Window=15)
- Symbolic Rule Engine
- Evaluation Pipeline

## 6. Remaining Limitations (Failure Modes)

### Failure Mode 1: Rare speakers absent from candidate set
- **Evidence:** EXP002b (92.76% ceiling). 
- **Current impact:** Regardless of downstream model quality, 7.24% of quotes are impossible to attribute correctly.
- **Potential future solution:** Learned semantic retrieval or expanding the candidate window dynamically based on scene boundaries.

### Failure Mode 2: Anaphoric References ("he said")
- **Evidence:** EXP003 (49.67% accuracy on Anaphoric quote types).
- **Current impact:** Deterministic rules cannot resolve pronouns to candidate identities, causing massive accuracy degradation when attribution is anaphoric.
- **Potential future solution:** Learned contextual representations (e.g., Coreference Resolution via ML).

### Failure Mode 3: Dialogue Continuation Ambiguity (A-B-A-B vs A-A)
- **Evidence:** EXP003 (31,605 rule conflicts between Dialogue Alternation and Previous Speaker).
- **Current impact:** The Rule Engine is forced to guess priority between a conversation continuing back-and-forth, or a single speaker talking across multiple quotes.
- **Potential future solution:** Learned discourse features that capture semantic flow (e.g., questions usually prompt answers).

## 7. Lessons Learned
- **Evaluation over engineering:** Measuring *applicability* vs *fired* exposed that Explicit attribution doesn't fail because the rule is wrong, but because the candidate generator drops the exact string representation.
- **Oracle testing:** Knowing the Rule Oracle Accuracy (76.26%) prevented us from wasting time trying to optimize priority ordering, as the rules themselves are the bottleneck.

## 8. Recommendations for Phase 2
Proceed immediately to Phase 2 (Machine Learning). The baseline is established at 66.53%. To surpass the 76.26% Rule Oracle ceiling, we must transition from deterministic rule ordering to a learned ranking model capable of synthesizing discourse history, contextual embeddings, and candidate features.
