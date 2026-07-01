# Speaker Attribution Research Project

## Master Documentation Index

---

# Purpose

This document serves as the central navigation guide for the entire research project.

It explains:

* the purpose of every major document,
* the dependency hierarchy between documents,
* the required reading order,
* which documents are authoritative,
* which documents are expected to evolve,
* and how research decisions should be made throughout the project.

This document intentionally contains **no implementation details** and **no research results**.

---

# Documentation Philosophy

This repository is organized as a layered research system rather than a collection of notes.

Information flows from high-level research goals down to implementation.

Lower-level documents must never contradict higher-level documents.

When uncertainty exists, always consult the document hierarchy before making changes.

---

# Documentation Hierarchy

The repository is divided into five logical layers.

```
Layer 0
MASTER_INDEX
(Project Navigation)

        ↓

Layer 1
Research Foundation

        ↓

Layer 2
Architecture

        ↓

Layer 3
Implementation Phases

        ↓

Layer 4
Experiments & Results
```

Information should always flow downward.

No document may redefine concepts established by a higher layer.

---

# Reading Order

Every new contributor or AI assistant should read the documents in the following order.

## Layer 1 — Research Foundation

1. 00_RESEARCH_CONSTITUTION.md
2. 01_PROJECT_VISION.md
3. 02_RESEARCH_QUESTIONS.md
4. 03_SUCCESS_CRITERIA.md
5. DESIGN_PHILOSOPHY.md
6. RESEARCH_GUARDRAILS.md
7. 04_DATASET.md
8. 05_EXPERIMENT_PROTOCOL.md
9. 06_EVALUATION_PROTOCOL.md
10. 07_RESEARCH_HYPOTHESES.md
11. KNOWN_UNKNOWNS.md
12. ASSUMPTIONS.md
13. RISK_REGISTER.md
14. RESEARCH_ROADMAP.md

Only after understanding these documents should architecture documents be read.

---

## Layer 2 — Architecture

Read in order:

1. SYSTEM_OVERVIEW
2. DISCOURSE_STATE
3. CANDIDATE_RANKING
4. MODEL_DESIGN

These documents describe what is being built.

They do not redefine research goals.

---

## Layer 3 — Implementation

Read only the active phase document.

Examples:

Phase 0

↓

Phase 1

↓

Phase 2

↓

...

Only one implementation phase should be considered active at a time.

Future phases are planning documents.

Completed phases become historical records.

---

## Layer 4 — Experiments

Experiments are never read before understanding the architecture.

Every experiment must reference:

* hypothesis
* implementation phase
* architecture version
* evaluation protocol

---

# Authority Order

If two documents appear to disagree, use the following precedence.

```
MASTER_INDEX

↓

RESEARCH_CONSTITUTION

↓

PROJECT_VISION

↓

RESEARCH_QUESTIONS

↓

DESIGN_PHILOSOPHY

↓

RESEARCH_GUARDRAILS

↓

ARCHITECTURE

↓

PHASE DOCUMENTS

↓

IMPLEMENTATION NOTES

↓

EXPERIMENT LOGS
```

Higher documents always override lower documents.

---

# Stable Documents

The following documents should change very rarely.

* MASTER_INDEX
* RESEARCH_CONSTITUTION
* PROJECT_VISION
* RESEARCH_QUESTIONS
* DESIGN_PHILOSOPHY

Changes to these documents fundamentally alter the direction of the project and should only occur after careful consideration.

---

# Living Documents

The following documents are expected to evolve.

* Architecture documents
* Phase documents
* Experiment logs
* Research log
* Decision records (docs/decisions/)
* Known unknowns
* Assumptions register
* Risk register
* Research roadmap

These documents should evolve as new evidence becomes available.

---

# Decision Process

Every significant modification should follow this workflow.

```
Question

↓

Hypothesis

↓

Experiment

↓

Evidence

↓

Decision

↓

Documentation

↓

Implementation
```

Implementation should never come before evidence.

---

# Research Rules

Before introducing any new model, feature, algorithm, dataset, or architectural component, answer the following questions.

1. What problem does this solve?

2. What evidence suggests this problem exists?

3. How will success be measured?

4. What is the simplest possible solution?

5. Does this change align with the Research Constitution?

If any question cannot be answered, the proposal should not be implemented.

---

# AI Assistant Responsibilities

AI assistants are implementation partners, not research directors.

They must:

* follow the document hierarchy,
* avoid introducing unnecessary complexity,
* preserve existing research goals,
* explain proposed changes before implementing them,
* document important decisions,
* ask for clarification rather than making assumptions.

AI assistants must not redefine the research question or alter evaluation methodology without explicit approval.

---

# Project Workflow

The expected workflow throughout the project is:

```
Research Question

↓

Architecture

↓

Implementation

↓

Experiment

↓

Evaluation

↓

Error Analysis

↓

Evidence

↓

Iteration
```

Iteration should always be driven by experimental evidence rather than intuition.

---

# Long-Term Goal

The objective of this repository is not merely to produce an accurate speaker attribution model.

The objective is to build a reproducible research framework that explains:

* what contextual information matters,
* why it matters,
* how much of it is required,
* and how lightweight systems can exploit it effectively.

Every document, experiment, and implementation should contribute toward answering those questions.
