---
Status: Frozen
Version: 1.0
Last Updated: 2026-07-02
Depends On: capability_matrix.md (v1.0)
Supersedes: None
---

# EXP010: Representation Selection Report

This document serves as the final decision gate before the implementation of EXP011. It evaluates the candidate representations for the top-priority reasoning capabilities identified in the `capability_matrix.md`, selecting the simplest justified approach for each.

## Evidence Traceability Table

Every architectural decision must be explicitly linked to experimental evidence from EXP010.

| Capability | EXP010 Frequency | Context Heatmap Evidence | Feasibility Assessment | Selected Representation |
| :--- | :--- | :--- | :--- | :--- |
| **Speaker Continuity** | 56.5% | Heavily clustered in Nearby/Conversation (0% Scene) | 85.8% Lightweight Feasible | Explicit Dialogue Memory (State Tracker) |
| **Coreference** | 19.5% | Local/Nearby | 94.9% Lightweight Feasible | Heuristic Coreference (Deferred to later experiment) |
| **Alias Matching** | 9.0% | Local | 100.0% Lightweight Feasible | Static Alias Dictionary (Deferred to later experiment) |
| **Pragmatics** | 15.0% | Conversation/Nearby | 50.0% Lightweight Feasible | Neural/Semantic Embedding (Deferred, absolute last resort) |

---

## 1. Selected Capability for EXP011: Speaker Continuity

Based on the Priority Matrix and Traceability Table, Speaker Continuity is the most urgent failure mode to address.

### Candidate Representation: Explicit Dialogue Memory (State Tracker)
* **Description:** A lightweight state tracker that persists the `previous_speaker` and `last_speaker` across sequential dialogue turns, enabling rule-based recovery of A-B-A-B alternations and interrupted turns.
* **Why this representation?** Speaker Continuity errors are fundamentally sequential. An explicit dialogue memory directly models the turn-taking structure of the conversation, perfectly aligning with the "Nearby" and "Conversation" context windows empirically identified in the heatmap.

### Alternatives Comparison
* **Alternative 1: N-gram Context Extension**
  * *Description:* Simply expand the local character search window backward by N sentences.
  * *Why rejected?* It does not distinguish between a character merely being *mentioned* in the narrative vs. a character actually participating in a dialogue turn.
* **Alternative 2: Neural Discourse Parser**
  * *Description:* A transformer model trained to predict conversational graph structures.
  * *Why rejected?* Violates the "simplest possible alternative" rule. The feasibility matrix indicates 85.8% of these errors can be solved with explicit features; deep semantics are unjustified at this stage.

* **Computational Cost of Selected:** Negligible. State tracking is O(1) per quote and requires no neural inference.
* **Expected Improvement:** High potential impact (bounds up to the 56.5% frequency limit).
* **Evaluation Criteria:** Accuracy gain in EXP011, measured specifically on the subset of quotes annotated as `Discourse: Speaker Continuity` in EXP010.

---

## 2. Deferred Capabilities (For Future Experiments)

To maintain a strict "one responsibility per experiment" philosophy, the other representations are deferred until EXP011 is fully executed and ablated. Each will undergo this exact selection process before their respective experiments.
