# ADR 002: Refined Candidate Generation (Window Size Expansion)

**Date:** 2026-07-01
**Status:** Accepted
**Supersedes:** ADR 001

## Context
Following EXP002, candidate generation was identified as the primary bottleneck for Phase 1 (capped at 88.29% recall). We conducted an ablation experiment (EXP002b) to test simple deterministic heuristics:
- **V1 (Alias Normalization):** Using `character_info.csv` to map aliases.
- **V2 (Window = 15):** Tracking the last 15 speakers instead of 5.
- **V3 (Window = 30):** Tracking the last 30 speakers.

## Results
| Variant | Recall | Avg Set Size | Efficiency |
|---------|--------|--------------|------------|
| Baseline (V0) | 88.29% | 3.65 | 24.22% |
| V1 (Alias) | 88.32% | 3.83 | 23.06% |
| V2 (Window 15) | 92.76% | 4.34 | 21.40% |
| V3 (Window 30) | 94.23% | 5.20 | 18.11% |

**Analysis:**
Alias normalization yielded negligible improvement (+0.03% recall), proving that aliases are not the primary failure mode. Expanding the discourse window to 15 quotes yielded a massive **+4.47% absolute recall increase** while keeping the average candidate set size manageable (4.34 candidates, adding less than 1 candidate per quote on average). 

## Decision
We will adopt **V2 (Window = 15)** as our standard heuristic and **permanently freeze candidate generation for Phase 1.**

## Rationale
V2 successfully pushes candidate recall into the mid-90s (92.76%) without violating the deterministic constraints of Phase 1 or introducing complex ML infrastructure. While the set size increased slightly over the strict 15% target (18.9% increase), adding <1 candidate on average is an extremely favorable trade-off for a nearly 5% absolute gain in theoretical upper-bound accuracy.

## Consequences
- The absolute theoretical ceiling for our Symbolic Attribution Baseline is now **92.76%**.
- All subsequent experiment runners (`run_exp003.py`, `run_exp004.py`) must pass the last 15 speakers to `previous_participants`.
