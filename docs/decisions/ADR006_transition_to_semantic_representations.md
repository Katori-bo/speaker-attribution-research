# ADR-006

## Title
Transition to Semantic Representations

## Context
Phase 3 (EXP009, EXP010, EXP011) rigorously evaluated the impact of explicit conversation-state memory (e.g., Participant Stack, turn tracking). The results demonstrated a performance plateau. While structural discourse features provide some baseline context, they proved insufficient for resolving the dominant residual error categories, particularly those involving identity (Coreference, Alias Matching) and interaction (Speaker Continuity, Speaker-Addressee Reasoning).

## Decision
Future contextual improvements will target lightweight semantic representations, evaluating them one capability at a time. The project will no longer pursue expanded structural state tracking as the primary means of improving attribution.

## Consequences
- No further explicit discourse-state features will be added to the baseline architecture unless new experimental evidence justifies it.
- Implementation will strictly follow a new hierarchy: `Failure Category → Computational Capability → Representation → Algorithm → Features`.
- The immediate focus shifts to resolving entity references via deterministic coreference chains (Phase 4).

## Supersedes
This ADR supersedes the earlier working assumption (from Phase 1 and 2) that richer, continuous conversation-state tracking was the primary direction for the research architecture. The architecture has officially pivoted to targeted semantic reasoning modules.
