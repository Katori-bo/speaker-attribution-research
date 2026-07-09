---
Status: Draft
Version: 1.1
Last Updated: 2026-07-02
Depends On: results/EXP010/capability_matrix.md (v1.1)
Supersedes: docs/architecture/representation_specification.md (v1.0)
---

# Architecture: Representation Specification

This document defines how missing contextual reasoning capabilities will be represented structurally, entirely independent of implementation details or code. 

## 1. Conversation State Module (Target: Speaker Continuity)

### Purpose
To track and maintain the atomic states of conversational participants across sequential dialogue turns, enabling the precise recovery of interrupted turns and A-B-A-B turn alternation without relying on neural embeddings.

### Atomic Decomposition
To determine the minimal effective representation for Speaker Continuity, the module decomposes "Conversation State" into independent atomic variables:

| Capability Focus | Atomic Representation | Rationale |
| :--- | :--- | :--- |
| Speaker Continuity | `previous_speaker` | Core variable for resolving A-B-A-B turn alternations. |
| Speaker Continuity | `current_speaker_persistence` | Core variable for resolving continuous A-A speech over multiple quotes. |
| Speaker Continuity | `dialogue_turn_index` | Tracks sequence depth to identify interruptions or sustained conversations. |
| Speaker Continuity | `conversation_boundary` | Tracks scene/paragraph breaks to prevent state leaking across distinct conversations. |
| Speaker Continuity | `active_conversation_id` | Unique ID grouping quotes to analyze multi-party vs dual-party dynamics. |

### Dependencies
- Baseline explicit quote attribution rules (to seed the state tracker)
- Scene/Paragraph boundary detection (to seed `conversation_boundary`)

### Required Inputs
- Current quotation boundary (start/end index)
- Candidate list for the current text window
- Identified local speakers (from baseline explicit rules)
- Scene/Paragraph boundaries

### Produced Outputs
- Updated atomic discourse state
- Derived features: `is_turn_alternation`, `is_interrupted_turn`, `expected_speaker`

### Internal State
- `last_speaker` (The entity who spoke the immediately preceding quote)
- `previous_speaker` (The entity who spoke the quote before `last_speaker`)
- `dialogue_turn_count` (Number of sequential quotes without a scene break)
- `conversation_id` (Identifier to group related quotes)

### Public Interface
- `initialize(scene_id)`: Resets state for a new narrative scene.
- `update(quote, explicit_speaker_prediction)`: Updates the state tracking based on the most recent quote.
- `export_features()`: Exports the derived contextual features to augment the baseline without modifying its downstream ranking mechanism.

### Update Rules
After processing a quotation, if a speaker is identified (via explicit rules or memory inference):
- `previous_speaker` ← `last_speaker`
- `last_speaker` ← `identified_speaker`
- `dialogue_turn_count` += 1

If a scene boundary is encountered:
- `last_speaker` ← None
- `previous_speaker` ← None
- `dialogue_turn_count` ← 0
- `conversation_id` += 1

### Known Limitations
- **Multi-party dialogue:** Assumes predominantly A-B-A-B structure. Will fail in chaotic conversations with 3+ active participants where turn order is non-deterministic.
- **Narrative interruptions:** Long blocks of narrative might artificially space out quotes, but the memory strictly counts *quotes*, not words. This may incorrectly bridge quotes that are separated by days of story time if no scene break is detected.

### Complexity Analysis
- **Time Complexity:** O(1) per quotation.
- **Memory Complexity:** O(1) space (storing exactly two entity references and an integer).
- **Interpretability:** Perfect. The state can be logged and audited at every step, allowing granular ablation of each atomic variable.
- **Maintenance Cost:** Extremely low. Relies on simple state transitions rather than trained parameters.
