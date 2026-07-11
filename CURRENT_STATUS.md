# Current Phase:
Phase 3 – Neural Sequence Modeling

## Completed Phases:
✓ Phase 0 – Project Foundation
✓ Phase 1 – Rule-based Baseline System
✓ Phase 2 – Initial Neural Feasibility & Feature Extraction

## Current Objective:
Iteratively evaluating feed-forward (MLP) and recurrent (GRU) neural architectures to isolate which context and memory features provide genuine architectural gains for speaker attribution.

## Next Milestone:
Solidifying the core state-free representation and establishing a robust baseline before re-introducing multi-turn contextual memory or entity tracking.

## Recent Experiments & Capabilities:

| Capability / Investigation | Experiment | Status |
| -------------------------- | ---------- | --------- |
| Neural Feasibility         | EXP019A    | Accepted |
| Neural Sequence Ceiling    | EXP020A    | Accepted |
| MLP Baseline (State-free)  | EXP021A    | Accepted (~72% Acc) |
| Relational Speaker GRU     | EXP022A    | Initially promising |
| Entity Anchor Binding      | EXP023     | Rejected (Memory instability) |
| Position/Order Salience    | EXP024     | Rejected (Dataset artifact) |
| GRU Memory Ablation        | EXP025     | Rejected (Fluke/No recurrent gain) |

## Active Focus:
Following **EXP025**, we discovered that the apparent gains from the `RelationalSpeakerGRU` were driven entirely by the candidate encoder acting on static features, with the recurrent memory cell acting merely as noise. We have pivoted back to treating the **NoMemoryEntityScorer (MLP)** as our primary, most robust baseline, avoiding recurrent complexity until we have stronger foundational features.
