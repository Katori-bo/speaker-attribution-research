# Phase 4A: Capability Decomposition

## Goal
The failure taxonomy established in EXP010 identified four dominant residual error categories: Coreference, Alias Matching, Speaker Continuity, and Speaker–Addressee Reasoning. 

This document decomposes those broad failure categories into specific computational capabilities required to resolve them.

To maintain methodological clarity, this phase strictly distinguishes between the following concepts:
`Failure Category → Computational Capability → Representation → Algorithm → Features`

This document focuses only on the first two steps: isolating the **Computational Capability** missing for each **Failure Category**.

## 1. Coreference
- **Failure Category:** Coreference failures
- **Computational Capability:** Resolve entity references

Coreference requires resolving different types of linguistic references to their underlying entities.

| Subtype | Example | Capability Required | Difficulty |
| :--- | :--- | :--- | :--- |
| Pronoun resolution | "He sighed." | Mapping gendered/plural pronouns to recent entities | Low |
| Possessive reference | "His wife arrived." | Resolving ownership/relation to an entity | Medium |
| Nominal reference | "The doctor nodded." | Mapping generic noun phrases to specific character roles | Medium |
| Bridging reference | "The old man left." | Inferring identity from semantic properties | High |

## 2. Alias Matching
- **Failure Category:** Alias Matching failures
- **Computational Capability:** Normalize entities across naming variations

Characters are frequently referred to by multiple names, titles, or relational markers.

| Subtype | Example | Capability Required | Difficulty |
| :--- | :--- | :--- | :--- |
| Name variants | "John" vs. "John Smith" | Sub-string matching and name normalization | Low |
| Title variants | "Mr. Smith" | Title recognition and canonical mapping | Low |
| Professional titles | "The Captain" | Mapping roles to specific characters in a scene | Medium |
| Nicknames | "Chief" | Contextual alias resolution | High |

## 3. Speaker Continuity
- **Failure Category:** Speaker Continuity failures
- **Computational Capability:** Track dialogue threads and active participants

When explicit conversation state fails to track participants, it is usually because the structural cues (like alternating turns) break down.

| Subtype | Example | Capability Required | Difficulty |
| :--- | :--- | :--- | :--- |
| Interrupted conversations | Dialogue broken by a long descriptive paragraph | Tracking conversation threads across narrative blocks | Medium |
| Multi-party tracking | Three or more characters speaking | Maintaining a dynamic local participant pool | High |
| Scene persistence | Characters moving between rooms | Tracking spatial/temporal bounds of a conversation | High |

## 4. Speaker–Addressee Reasoning
- **Failure Category:** Speaker–Addressee Reasoning failures
- **Computational Capability:** Infer conversational roles and targets

Resolving "who is speaking" sometimes requires knowing "who is being spoken to."

| Subtype | Example | Capability Required | Difficulty |
| :--- | :--- | :--- | :--- |
| Directed dialogue | "What do you think, John?" | Vocative extraction and target identification | Medium |
| Implicit addressee | (A question answered by B) | Conversational adjacency pair mapping | High |
| Third-party references | "Tell him I said no." | Distinguishing addressees from mentioned entities | High |
