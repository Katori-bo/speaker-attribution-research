# Phase 4B: Representation Survey

## Goal
For each decomposed computational capability identified in Phase 4A, this document surveys the lightest possible semantic representation that could solve it. 

Following the project's strict hierarchy (`Capability → Representation → Algorithm`), this document defines how the required information should be structured (the Representation) and how it might be extracted (the Algorithm), rather than proposing neural models directly.

## Candidate Representations

### 1. Coreference
- **Computational Capability:** Resolve entity references
- **Candidate Representation:** Coreference chains (linked mention spans to canonical IDs)
- **Algorithm:** Preprocessing pipeline (e.g., BookNLP)
- *Rationale:* Highly interpretable and sufficient for resolving explicit pronoun/nominal links.

### 2. Alias Matching
- **Computational Capability:** Normalize entities across naming variations
- **Candidate Representation:** Canonical Alias Dictionary (static lookup table mapping aliases to a canonical ID)
- **Algorithm:** Heuristic alias clustering or NLP pipeline character lists
- *Rationale:* Extremely lightweight, highly interpretable, and resolves the majority of static alias mismatches computationally for almost free.

### 3. Speaker Continuity
- **Computational Capability:** Track dialogue threads and active participants
- **Candidate Representation:** Conversation Graph (linking dialogue turns based on temporal adjacency and semantic response)
- **Algorithm:** Rule-based heuristics over adjacency pairs and vocatives
- *Rationale:* Explicitly models the thread of dialogue over interruptions, offering structure beyond linear stacks.

### 4. Speaker–Addressee Reasoning
- **Computational Capability:** Infer conversational roles and targets
- **Candidate Representation:** Local Dialogue Graph (directed graph within a scene where nodes are characters and edges are directed utterances)
- **Algorithm:** Rule-based heuristics combined with NLP dependency parsing
- *Rationale:* Captures the interaction dynamics required to untangle multi-party scenes.

## Next Steps
Before implementation, these candidate representations must undergo a strict Representation Justification Audit (Phase 4C) to verify novelty, feasibility, and expected impact.
