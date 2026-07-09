---
Status: Draft
Version: 1.1
Last Updated: 2026-07-02
Depends On: 
  - docs/architecture/ADR002_representation_selection.md (v1.0)
  - results/EXP010/capability_matrix.md (v1.1)
Supersedes: docs/architecture/EXP011_design.md (v1.0)
---

# EXP011 Design: Conversation State Evaluation

This document completely specifies EXP011 before implementation begins. To ensure maximum interpretability of what information is actually necessary for Speaker Continuity, the experiment is split into two phases: Representation Evaluation (EXP011A) and Feature Ablation (EXP011B).

---

## Part 1: EXP011A - Conversation State Representation

### Research Question
Can an explicit conversation state representation recover Speaker Continuity errors without utilizing semantic embeddings?

### Hypothesis
By augmenting the frozen representation with exported contextual features (such as previous speaker and turn indices) while leaving the downstream attribution mechanism unchanged, the system will significantly recover Speaker Continuity failures.

### Experimental System
- Baseline: Frozen EXP009 explicit rule-based system.
- Experimental: Baseline augmented by `ConversationStateModule` output features.

### Independent Variable
- `ConversationStateModule` (Active vs Inactive)

### Dependent Variables (Metrics for both A and B)
- Global Accuracy, Precision, Recall, F1
- Runtime (Quotes per second) & Memory (Peak RAM utilization)
- **Recovered Speaker Continuity Errors**
- **New Errors Introduced** (Degradation on previously correct quotes)
- **Net Gain** 

### Expected Observations
- Speaker Continuity errors decrease significantly.
- Coreference and Alias Matching errors remain largely unchanged.
- Runtime overhead remains negligible due to O(1) processing.
- A small margin of new errors may be introduced due to multi-party conversations confusing the A-B-A-B state.

---

## Part 2: EXP011B - Conversation State Feature Ablation

### Research Question
Which specific atomic variables within the conversation state actually contribute to the recovery of Speaker Continuity?

### Hypothesis
The majority of the gain from explicit conversation state is attributable to tracking speaker persistence (`last_speaker`) and dialogue turn alternation (`previous_speaker`), while conversation boundary identifiers contribute marginally.

### Ablation Sequence
Each variable will be augmented to the baseline individually to isolate its impact:
1. Baseline + `previous_speaker` feature ONLY.
2. Baseline + `dialogue_turn_index` feature ONLY.
3. Baseline + `conversation_boundary` feature ONLY.
4. Baseline + FULL `ConversationStateModule`.

### Expected Observations
- `previous_speaker` will account for the vast majority of recovered Speaker Continuity errors.
- `conversation_boundary` will have a negligible impact on accuracy but will prevent long-range cascading false positives across scenes.

---

## Evaluation Workflow (for both A and B)
1. Baseline Evaluation (Establish EXP009 metrics on the target PDNC test set).
2. Experimental Evaluation (Run exactly the same test set with augmented features).
3. Metric Extraction (Compute global metrics and Net Gain/Loss).
4. Secondary Evaluation (Run category analysis on EXP010 subset annotations to isolate *why* accuracy changed).
5. Interpret Results.
