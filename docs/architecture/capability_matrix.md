---
Status: Draft
Version: 1.0
Last Updated: 2026-07-02
Depends On: results/EXP010/capability_matrix.md (v1.1)
Supersedes: None
---

# Architecture: Capability Matrix

This document maps the empirically missing capabilities (from EXP010) to their candidate architectural representations and indicates which representation has been selected for implementation.

## Candidate Representation Mapping

| Capability | Candidate Representations | Selected |
| :--- | :--- | :--- |
| **Speaker Continuity** | Dialogue Memory<br>N-gram Context Extension<br>Conversation Graph (Neural) | **Dialogue Memory** |
| **Pronominal Coreference** | Heuristic Coreference<br>Neural Coreference Resolver | Deferred |
| **Alias Matching** | Static Alias Dictionary<br>Fuzzy String Matching | Deferred |
| **Speaker-Addressee Pragmatics** | Conversational Role Graph<br>Contextual Embedding (RoBERTa) | Deferred |
