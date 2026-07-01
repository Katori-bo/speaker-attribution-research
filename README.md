# Speaker Attribution Research

## Overview

This repository is a research project investigating whether lightweight models with explicit contextual representations can achieve competitive speaker attribution accuracy in literary novels — without relying on large language models.

The project is structured as a layered, evidence-driven research framework. Every architectural decision, contextual feature, and model component is introduced incrementally and justified by experimental evidence.

---

## Research Motivation

Speaker attribution — identifying who is speaking in a novel — is typically treated as either a rule-based problem or as a subtask for large language models. Both approaches have limitations:

* Rule-based systems struggle with implicit and ambiguous quotations.
* LLM-based systems are expensive, opaque, and difficult to reproduce.

This project explores a middle path: building a lightweight, interpretable system that uses explicit contextual features (speaker memory, active characters, recent mentions) to perform accurate speaker attribution.

The goal is understanding **which contextual information matters and how much of it is required**, not simply maximizing benchmark accuracy.

---

## Research Questions

**Primary:** Can lightweight contextual representations recover most of the performance of much larger models for literary speaker attribution?

**Secondary:**

* Which contextual signals matter most?
* How much discourse memory is required?
* Does candidate ranking outperform direct classification?
* When do additional contextual features stop helping?
* Does an LLM teacher improve a strong lightweight baseline?

Full details: [02_RESEARCH_QUESTIONS.md](docs/02_RESEARCH_QUESTIONS.md) and [07_RESEARCH_HYPOTHESES.md](docs/07_RESEARCH_HYPOTHESES.md)

---

## Repository Organization

```
speaker-attribution-research/
│
├── MASTER_INDEX.md          ← Central navigation guide
├── CURRENT_STATUS.md        ← Entry point for every session
├── README.md                ← This file
│
├── docs/
│   ├── 00–06_*.md           ← Research Foundation (Layer 1)
│   ├── 07_RESEARCH_HYPOTHESES.md
│   ├── DESIGN_PHILOSOPHY.md
│   ├── RESEARCH_GUARDRAILS.md
│   ├── KNOWN_UNKNOWNS.md
│   ├── ASSUMPTIONS.md
│   ├── RISK_REGISTER.md
│   ├── RESEARCH_ROADMAP.md
│   │
│   ├── architecture/        ← System Architecture (Layer 2)
│   ├── phases/              ← Implementation Phases (Layer 3)
│   ├── decisions/           ← Architecture Decision Records
│   └── prompts/             ← AI Agent Operating Rules
│
├── src/                     ← Source code
├── experiments/             ← Experiment logs and results
├── results/                 ← Final evaluation outputs
├── data/                    ← Datasets
├── notebooks/               ← Exploratory analysis
├── scripts/                 ← Utility scripts
├── templates/               ← Report and log templates
└── logs/                    ← Runtime logs
```

---

## Documentation Guide

The repository follows a strict documentation hierarchy. Read documents in the order specified below.

| Layer | Documents | Purpose |
|-------|-----------|---------|
| 0 | [MASTER_INDEX.md](MASTER_INDEX.md) | Central navigation and authority order |
| 1 | [Research Foundation](docs/) | Research goals, questions, constraints |
| 2 | [Architecture](docs/architecture/) | System design and component boundaries |
| 3 | [Phases](docs/phases/) | Implementation execution plans |
| 4 | [Experiments](experiments/) | Experiment logs and results |

**Start here:** [MASTER_INDEX.md](MASTER_INDEX.md)

Lower-level documents must never contradict higher-level documents.

---

## Current Project Status

See [CURRENT_STATUS.md](CURRENT_STATUS.md) for the live status of the project.

---

## Development Workflow

1. Read `CURRENT_STATUS.md` to identify the active phase.
2. Read `MASTER_INDEX.md` for the full documentation hierarchy.
3. Read the active phase document in `docs/phases/`.
4. Implement according to the phase's deliverables and acceptance criteria.
5. Document decisions in `docs/decisions/`.
6. Log experiments in `experiments/`.

---

## Experiment Workflow

Every experiment follows this process:

```
Hypothesis → Design → Implementation → Evaluation → Error Analysis → Documentation
```

1. State which hypothesis is being tested (see [07_RESEARCH_HYPOTHESES.md](docs/07_RESEARCH_HYPOTHESES.md)).
2. Follow the experiment protocol (see [05_EXPERIMENT_PROTOCOL.md](docs/05_EXPERIMENT_PROTOCOL.md)).
3. Evaluate using the evaluation protocol (see [06_EVALUATION_PROTOCOL.md](docs/06_EVALUATION_PROTOCOL.md)).
4. Record results using the experiment report template (see [templates/](templates/)).
5. Update relevant documents based on evidence.

---

## Getting Started

1. Read [MASTER_INDEX.md](MASTER_INDEX.md) for the complete documentation hierarchy and reading order.
2. Read [CURRENT_STATUS.md](CURRENT_STATUS.md) to understand the current project state.
3. Read the active phase document to understand the current objectives.
4. Review the AI operating rules in `docs/prompts/` if you are an AI assistant.

---

## Repository Philosophy

* **Evidence before implementation.** No component is added without experimental justification.
* **Simplicity before complexity.** Prefer the simplest solution with equivalent performance.
* **Understanding before accuracy.** The goal is explaining what works and why, not maximizing a leaderboard score.
* **Reproducibility is mandatory.** Every experiment must be repeatable.
* **Documents are authoritative.** The documentation hierarchy governs all decisions.
