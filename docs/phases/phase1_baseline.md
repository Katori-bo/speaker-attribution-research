# Phase 1 — Baseline System

## Objective

Establish simple, explainable baselines that define the minimum performance level every future system must exceed.

---

## Research Question

How much performance can be obtained without machine learning?

---

## Deliverables

* Dialogue extraction
* Candidate generation
* Rule-based attribution
* Baseline metrics

---

## Components

### Dialogue Detection

Extract dialogue spans.

### Explicit Attribution

Identify direct patterns such as:

Character said

said Character

Character replied

etc.

### Candidate Generation

Generate plausible speaker candidates.

### Heuristic Ranking

Simple rules only.

Examples:

* nearest mention
* previous speaker
* dialogue alternation
* speech verbs

---

## Evaluation

Run against PDNC.

Measure

* Accuracy
* Precision
* Recall
* F1

Perform error analysis.

---

## Acceptance Criteria

Reliable baseline.

Every prediction explainable.

No machine learning.

---

## Out of Scope

* Neural models
* Discourse state
* LLMs
