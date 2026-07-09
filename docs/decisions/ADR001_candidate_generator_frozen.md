# ADR 001: Freeze Candidate Generation for Phase 1

**Date:** 2026-07-01
**Status:** Superseded (Pending EXP002b)

## Reason for Superseding
Candidate recall (88.29%) was lower than expected, with severe degradation on rare speakers (44.49%). Experiment EXP002b has been authorized to investigate simple deterministic improvements before permanently freezing the generator. Final freeze is deferred until experiment completion.

## Context
During Phase 1 (Symbolic Baseline), we need a reliable candidate generation mechanism. We implemented a sliding-window heuristic that extracts explicit mentions and addressees directly from the PDNC annotations and includes the speakers of the last 5 quotes. 

In EXP002, we measured this candidate generator's Oracle Accuracy over the entire 37,131 quote corpus.

**Results:**
- Global Oracle Accuracy: 88.29%
- Recall for Frequent Speakers (>5 quotes): 89.10%
- Recall for Rare Speakers (<=5 quotes): 44.49%
- Average Candidate Set Size: 3.65

## Decision
We will **freeze** this candidate generation logic for the remainder of Phase 1. We accept the 88.29% Oracle Accuracy as the absolute ceiling (upper bound) for our symbolic attribution baseline. 

## Rationale
Candidate generation is a complex subproblem. Attempting to improve the 44.49% rare speaker recall using coreference resolution or wider context windows would constitute a machine learning problem, violating the constraints of Phase 1 (Deterministic/Symbolic only). 

By freezing the generator at 88.29% recall with a tight candidate set size of 3.65, we have a perfectly stable foundation to evaluate the deterministic attribution rules (EXP003 / EXP004). If the deterministic rules achieve an accuracy of, say, 85%, we will know they are performing exceptionally well relative to the 88.29% ceiling.

## Consequences
- 11.71% of all quotes in Phase 1 will inevitably be attributed incorrectly because the true speaker is not in the candidate set.
- Rare speakers are disproportionately penalized by this baseline architecture.
- Improving candidate generation becomes a primary objective for Phase 2.
