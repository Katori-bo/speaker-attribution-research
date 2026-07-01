# Model Design

## Purpose

This document describes the learning model at a conceptual level.

Specific architectures will be explored experimentally.

No model is permanently fixed.

---

## Core Philosophy

The project does not seek the largest model.

The objective is identifying the smallest model capable of solving the task effectively.

---

## Model Inputs

Dialogue text

Candidate representation

Context representation

Local linguistic features

Optional structural features

---

## Model Output

For every candidate:

Probability(candidate is speaker)

Final prediction:

Highest probability candidate.

---

## Initial Baseline

The first trainable model should remain intentionally simple.

The purpose is establishing a reliable baseline before introducing additional complexity.

---

## Model Evolution

Future models may include:

* Lightweight transformer encoders
* Recurrent sequence models
* Feature-based machine learning models
* Hybrid neural-feature models

The choice must be justified experimentally.

---

## Model Constraints

Models should prioritize:

Interpretability

Efficiency

Reproducibility

Low memory usage

Fast inference

---

## Research Strategy

Model complexity increases only after:

* Baselines are established.
* Error analysis identifies limitations.
* A proposed improvement has a measurable hypothesis.

---

## Success Definition

The objective is not achieving the highest benchmark score.

Success is defined as understanding which contextual representations contribute most to accurate speaker attribution while maintaining a lightweight and interpretable system.

The architecture should remain modular so that future research can replace individual components without redesigning the entire pipeline.
