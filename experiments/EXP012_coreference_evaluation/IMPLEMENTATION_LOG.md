# EXP012 Implementation Log

| Stage     | Status      | Date       | Notes |
| --------- | ----------- | ---------- | ----- |
| EXP012A.0 | COMPLETED   | 2026-07-02 | Data Source Validation (coverage, precision, feasibility). |
| EXP012A.1 | COMPLETED   | 2026-07-02 | Annotation Import (built BookNLP parser & mapping layer). |
| EXP012A.2 | COMPLETED   | 2026-07-02 | Representation Validation (alignment, tests, integrity checks). |
| EXP012A.3 | COMPLETED   | 2026-07-02 | Semantic Feature Extraction (isolated exactly four coreference capabilities). |
| EXP012A.3b | COMPLETED   | 2026-07-02 | Alias Representation Extension (parsed .book, achieved 99.7% mapping). |
| EXP012A.4 | COMPLETED   | 2026-07-02 | Integration (added feature provider and integration tests). |
| EXP012A.5 | COMPLETED   | 2026-07-02 | Evaluation (demonstrated +2.04% accuracy gain on PrideAndPrejudice). |

## Engineering Constraints
> **Constraint:** No modifications to the existing EXP011 feature extraction code. 
> EXP012 introduces an additional feature provider that is toggled via experiment configuration.

---

## Detailed Log

### EXP012A.0 Data Source Validation
- **Status:** COMPLETED
- **Date:** 2026-07-02
- **Metrics Collected:**
  - Quote Coverage: 65.33% (1,159 / 1,774)
  - Character Participation: 100.00%
  - Pronoun Resolution: 64.19% (10,587 / 16,493)
- **Feature Feasibility:** All four required features are computable from `.entities` and `.quotes`.
- **Integration Risk:** Token alignment discrepancy between BookNLP `.tokens` and PDNC byte offsets requires mapping.
- **Archive:** `archives/EXP012A.0/` (contains `validation_report.md`, `STATUS.md`, `metadata.json`, `environment.md`, `booknlp_pytorch_compat.patch`)

### EXP012A.1 Annotation Import
- **Status:** COMPLETED
- **Date:** 2026-07-02
- **Notes:** 
  - Created `src/coreference/schemas.py`, `src/coreference/parser.py`, and `src/coreference/mapping.py` to reconstruct canonical mention-to-entity mapping from BookNLP output. 
  - Restricted scope explicitly to parsing (no features or ranking logic).
  - Built comprehensive unit tests (`tests/test_coreference_parser.py`) verifying file malformations, overlaps, missing IDs, and invalid offsets.
- **Archive:** `archives/EXP012A.1/` (contains `STATUS.md`, `metadata.json`)

### EXP012A.2 Representation Validation
- **Status:** COMPLETED
- **Date:** 2026-07-02
- **Notes:** 
  - Validated coreference chains (no invalid chains, order preserved).
  - Built `src/coreference/alignment.py` to map PDNC byte offsets to BookNLP token IDs.
  - Handled tabular offset misalignments in BookNLP tokens resulting from literal strings.
  - Ran `scripts/validate_representation.py`, achieving 100% alignment rate on 1,270 quotes.
  - Built comprehensive tests in `tests/test_coreference_validation.py` and `tests/test_coreference_alignment.py`.
- **Archive:** `archives/EXP012A.2/` (contains `STATUS.md`, `metadata.json`, `representation_statistics.md`)

### EXP012A.3 Semantic Feature Extraction
- **Status:** COMPLETED
- **Date:** 2026-07-02
- **Notes:** 
  - Isolated exactly four coreference capabilities based on EXP012 design constraints.
  - Features defined: `candidate_in_quote_chain`, `nearest_coref_dist` (absolute), `recent_mention_count` (50-token window), and `chain_recency`.
  - Achieved full determinism with passing unit tests.
  - Correlation audit verified feature orthogonality (low collinearity).
- **Archive:** `archives/EXP012A.3/` (contains `STATUS.md`, `metadata.json`, `feature_dictionary.md`)

### EXP012A.3b Alias Representation Extension
- **Status:** COMPLETED
- **Date:** 2026-07-02
- **Notes:**
  - Diagnosed initial evaluation failure: mapped 0% of candidates because of strict lexical substring matches against sub-tokens.
  - Justified `.book` extraction strictly as an alias mapping layer.
  - Re-wrote MentionToEntityMapper to use `.book` alias dictionary.
  - Validated mapping: reached 99.70% mapping coverage across 7,382 candidates on PrideAndPrejudice.

### EXP012A.4 Integration
- **Status:** COMPLETED
- **Date:** 2026-07-02
- **Notes:** 
  - Implemented `SemanticFeatureProvider` in `src/coreference/pipeline.py` which isolates semantic feature extraction.
  - Implemented `scripts/run_exp012.py` as a configuration-controlled pipeline connecting Baseline (EXP011) features vs Experimental (EXP012 Semantic Features).
  - Maintained frozen conditions for candidates, splits, dataset, and baseline features.
  - Created and passed a deterministic integration test (`tests/test_semantic_integration.py`).

### EXP012A.5 Evaluation
- **Status:** COMPLETED
- **Date:** 2026-07-02
- **Notes:**
  - Extracted Coreference semantic features.
  - Modified `run_exp012.py` to evaluate strictly on PrideAndPrejudice with a 5-fold `GroupKFold` cross-validation split, due to lack of parsed BookNLP outputs for other dataset novels.
  - Recovered 50 coreference failure cases; 24 new errors.
  - Overall accuracy improved from 0.7709 to 0.7913 (+2.04% absolute).
