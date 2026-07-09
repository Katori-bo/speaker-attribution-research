---
Status: Frozen
Version: 1.1
Last Updated: 2026-07-02
Depends On: scientific_interpretation.md (v1.0), semantic_annotations_master.csv
Supersedes: results/EXP010/capability_matrix.md (v1.0)
---

# EXP010: Empirical Capability Matrix

This document summarizes the empirical failure categories discovered in EXP010. It serves as pure evidence representing the current limitations of the explicit baseline architecture. No architectural representations or implementations are proposed in this document.

## Capability Evidence Table

| Capability ID | Capability Name | Frequency | Required Context Window | Annotator Judged Lightweight Feasible? |
| :--- | :--- | :--- | :--- | :--- |
| **CAP001** | **Speaker Continuity** | 56.5% | Nearby / Conversation | Yes (85.8%) |
| **CAP002** | **Pronominal Coreference** | 19.5% | Local / Nearby | Yes (94.9%) |
| **CAP003** | **Speaker-Addressee Pragmatics** | 15.0% | Nearby / Conversation | Mixed (50.0%) |
| **CAP004** | **Alias Matching** | 9.0% | Local | Yes (100.0%) |

## Failure Mechanisms

### CAP001: Speaker Continuity
The explicit baseline isolates quotes from their conversational context. It cannot track interrupted turns, A-B-A-B turn alternation, or implicit continuation where no explicit referring expression exists within the immediate text window.

### CAP002: Pronominal Coreference
The baseline relies on string matching near dialogue verbs. It cannot map pronouns ("she", "he") back to their antecedents in the preceding narrative.

### CAP003: Speaker-Addressee Pragmatics
The baseline cannot infer who is speaking based on internal semantic clues about who is being spoken *to* or *about*.

### CAP004: Alias Matching
The baseline requires exact or strict substring matches against a candidate list. It fails on relational aliases ("his wife") or morphological variations ("The King" vs "The King of Hearts").
