# Phase 4 — LLM Extension

## Objective

Determine whether an LLM provides measurable additional value beyond the lightweight model.

This phase is optional.

The project must remain complete even if this phase is skipped.

---

## Research Question

Does teacher supervision improve lightweight speaker attribution?

---

## Phase 4a

### Teacher Evaluation

Run teacher on PDNC.

Measure

* accuracy
* agreement with gold labels
* difficult quote performance

---

## Phase 4b

### Pseudo-labeling

Generate labels for additional novels.

Train student again.

Measure improvement.

---

## Phase 4c

### Soft Targets

Replace hard labels with probability distributions.

Evaluate.

---

## Phase 4d

### Reasoning Supervision

Only begin if earlier experiments justify it.

Investigate whether teacher reasoning improves contextual representations.

---

## Deliverables

Teacher evaluation.

Comparison with student.

Cost analysis.

Runtime comparison.

---

## Acceptance Criteria

Quantify exactly what the teacher contributes.

If teacher provides no measurable benefit, document that result.

---

## Out of Scope

Replacing the student with an LLM.

End-to-end LLM inference.
