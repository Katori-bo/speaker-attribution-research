---
Status: Draft
Version: 1.0
Last Updated: 2026-07-02
Depends On: 
  - results/EXP010/capability_matrix.md (v1.0)
  - docs/architecture/representation_specification.md (v1.0)
Supersedes: None
---

# ADR002: Representation Selection for Speaker Continuity

## Context
EXP010 revealed that Speaker Continuity is the dominant residual failure mode in our baseline explicit speaker attribution system, accounting for 56.5% of missing capabilities. To progress to EXP011, we must select an architectural representation that recovers this capability while preserving the project's commitment to interpretability and minimal computational overhead.

## Problem
The current explicit representation isolates quotations from their conversational context. It cannot track interrupted turns, A-B-A-B turn alternations, or implicit continuations where no explicit referring expression exists within the immediate text window.

## Evidence
- **Frequency:** Speaker Continuity accounts for 56.5% of EXP010 residual errors (EXP010 Table 1).
- **Context Window:** The heatmap reveals this capability heavily clusters in `Nearby` (50) and `Conversation` (46) windows, with 0 instances requiring `Scene`-level context.
- **Feasibility:** Annotators judged that 85.8% of these failures are recoverable using explicit lightweight rules without deep semantics.

## Alternative Representations

| Capability | Representation | Complexity | Interpretability | Runtime Cost | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Speaker Continuity | **Dialogue Memory** | Low | High | Low | **Selected** |
| Speaker Continuity | **N-gram Context Extension** | Low | Medium | Low | **Rejected** |
| Speaker Continuity | **Conversation Graph (Neural)** | High | Low | High | **Rejected** |

## Chosen Representation
**Dialogue Memory:** An explicit, O(1) state tracker that persists the `previous_speaker` and `last_speaker` across sequential dialogue turns.

## Rejected Representations
- **N-gram Context Extension:** Rejected because blindly expanding the search window backward fails to distinguish between a character being merely mentioned in the narrative versus a character actively participating in the dialogue turn.
- **Conversation Graph (Neural):** Rejected because it violates the "simplest possible alternative" rule. The feasibility matrix indicates deep semantics are unjustified for this specific capability.

## Risks
- The Dialogue Memory assumes an A-B-A-B structure. It may introduce false positives in multi-party conversations (3+ active participants) where turn order is non-deterministic.

## Future Alternatives
- If Dialogue Memory fails to recover the expected margin, or if false positives are too high in multi-party scenes, we may explore a hybrid approach that builds a local Conversation Graph without neural embeddings.

## Consequences
- The system will become stateful at the conversation level.
- Quotations must be processed sequentially within a scene, prohibiting parallelized, order-independent processing of quotes within the same document.

## Acceptance Criteria
- Meets all criteria defined in `representation_validation_plan.md` for Dialogue Memory.
