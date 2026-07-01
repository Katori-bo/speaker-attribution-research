# Phase 3 — Lightweight Student Model

## Objective

Develop a lightweight speaker attribution model trained directly on gold labels.

No LLM supervision is used during this phase.

---

## Primary Research Question

What is the minimal contextual representation required for accurate speaker attribution?

---

## Overall Strategy

Develop incrementally.

Only one contextual component should be introduced at a time.

Every addition requires experimental evidence.

---

## Phase 3a

### Candidate Ranking

No discourse state.

Inputs:

* quote
* local context
* candidate list

Output:

Candidate ranking.

Evaluate.

Record results.

---

## Phase 3b

### Add Speaker Memory

Introduce

* last speaker
* previous speaker

Measure improvement.

---

## Phase 3c

### Active Character Representation

Introduce

* active characters
* recent participation

Measure improvement.

---

## Phase 3d

### Recent Mention Representation

Introduce

* recent character mentions
* mention distance
* mention frequency

Measure improvement.

---

## Phase 3e

### Error Analysis

Study remaining failures.

Group errors.

Determine missing contextual information.

Do not add new components without evidence.

---

## Required Experiments

Every stage must include

* training
* evaluation
* ablation
* runtime
* memory usage

---

## Required Deliverables

State vs Accuracy table.

Error taxonomy.

Failure examples.

---

## Acceptance Criteria

Improves upon previous baselines.

Produces reproducible experiments.

Every contextual addition experimentally justified.

---

## Out of Scope

Teacher models.

Reasoning traces.

Pseudo-labeling.
