---
Status: Frozen
Version: 1.0
Last Updated: 2026-07-02
Depends On: semantic_annotations_master.csv, capability_statistics.csv
Supersedes: None
---

# EXP010: Scientific Interpretation

This document synthesizes the empirical evidence gathered during the EXP010 error taxonomy annotation (N=200). Its purpose is to interpret the quantitative findings and translate them into actionable, evidence-backed hypotheses regarding contextual reasoning in speaker attribution.

## 1. Discourse Tracking is the Principal Missing Capability

**Finding:** Speaker Continuity accounts for the majority (**56.5%**) of residual failures.

**Interpretation:** Over half of the residual attribution errors are not caused by lexical ambiguity or missing symbolic references within the immediate sentence. Rather, they are caused by a failure to maintain the discourse state across sequential dialogue turns (e.g., A-B-A-B turn alternation, interrupted turns, implicit continuation). This indicates that discourse tracking—not merely surface-level feature matching—is the principal missing reasoning capability of the evaluated explicit representation.

## 2. Most Failures Appear Recoverable via Lightweight Representation

**Finding:** The annotator feasibility assessment reveals that Alias Matching (100%), Pronominal Coreference (94.9%), and Speaker Continuity (85.8%) are overwhelmingly judged to be solvable without deep semantic modeling. Only Pragmatics presents significant resistance (50% solvable, 30% unsolvable without semantics, 20% unsure).

**Interpretation:** The evidence strongly suggests that deep, resource-heavy semantic models are largely unnecessary for the vast majority of attribution errors. Instead, most remaining failures appear recoverable through targeted, lightweight reasoning mechanisms. This validates the project's hypothesis that explicit, structural representations can capture the vast majority of speaker attribution phenomena if given the correct contextual framing.

## 3. Global Narrative Reasoning is Rarely Required

**Finding:** The context window heatmap demonstrates that Speaker Continuity failures heavily cluster in the `Nearby` (50) and `Conversation` (46) context windows, with exactly zero instances requiring `Scene`-level context. 

**Interpretation:** The resolution of speaker continuity rarely requires scene-level context or long-document reasoning. The required context is tightly localized to previous dialogue turns and immediate conversational memory. This suggests that any proposed representation for discourse tracking does not need to model the global narrative, but rather just the immediate, local conversational state.

## 4. Pragmatics Remains a Semantic Bottleneck

**Finding:** Pragmatics: Speaker–Addressee Semantics accounts for 15.0% of failures, and is the only category where a lightweight explicit fix was frequently deemed insufficient.

**Interpretation:** Recovering attribution via internal semantic clues (e.g., inferring the speaker based on who is being spoken *to* or *about*) fundamentally resists lightweight, explicit heuristics. While not the most frequent failure mode, Pragmatics represents the hard upper bound of what lightweight architectures can achieve.

## Summary Conclusion
The quantitative evidence supports the hypothesis that long-document reasoning and global semantic embeddings are largely unnecessary for speaker attribution. The primary bottleneck is the lack of a structured, local dialogue memory capable of preserving speaker state across adjacent conversational turns.
