# System Overview

## Purpose

This document describes the overall architecture of the research system.

It defines how information flows through the system and the responsibility of each major component. It intentionally avoids implementation details.

---

## System Philosophy

The system is designed around the belief that speaker attribution is primarily a contextual reasoning problem rather than a language generation problem.

Instead of relying on extremely large models to perform implicit reasoning, the system explicitly represents the contextual information necessary for attribution.

The architecture emphasizes:

* Explicit context representation
* Lightweight computation
* Modular components
* Incremental reasoning
* Interpretability
* Reproducibility

---

## Architectural Evolution

The project architecture evolves based on experimental evidence:

* **Stage 1: Deterministic features**
  Initial baseline relying on structural cues.
* **Stage 2: Explicit discourse state**
  Contextual caching of recent narrative events (frozen after EXP011).
* **Evidence: Performance plateau**
  EXP011 demonstrated that richer explicit state yields diminishing returns.
* **Stage 3: Minimal semantic representations**
  Current focus: targeting specific failure categories (e.g., coreference) with lightweight semantic modules.

---

## High-Level Pipeline

Raw Novel

↓

Dialogue Detection

↓

Candidate Generation

↓

Context Representation

↓

Candidate Ranking Model

↓

Speaker Prediction

↓

Evaluation

---

## Component Responsibilities

### Dialogue Detection

Responsible for locating dialogue spans within the novel.

Input:
Raw novel text

Output:
Sequence of dialogue segments.

---

### Candidate Generation

Responsible for generating the possible speakers for each dialogue.

Candidate generation should prioritize recall over precision.

Removing the correct speaker is considered a critical error.

---

### Context Representation

Maintains contextual information useful for speaker attribution.

Examples include:

* Active characters
* Recent mentions
* Dialogue history
* Local linguistic features

The context representation should remain lightweight and interpretable.

---

### Candidate Ranking

Assigns a probability score to every candidate.

The system predicts:

P(candidate | quote, context)

rather than performing fixed-label classification.

---

### Evaluation

Measures both prediction quality and computational efficiency.

Evaluation is defined separately in the evaluation protocol.

---

## Architectural Principles

The architecture follows the following rules:

* Components must remain modular.
* Components should be independently replaceable.
* Information flows in one direction.
* New components require experimental justification.
* Simpler architectures are preferred when performance is comparable.
