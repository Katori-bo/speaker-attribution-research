# Context Representation (Discourse State)

## Purpose

This document defines the contextual information maintained while processing a novel.

The objective is not narrative understanding.

The objective is maintaining sufficient context for speaker attribution.

---

## Design Philosophy

The discourse state is a structured cache rather than a world model.

It stores only information directly relevant to speaker attribution.

---

## Guiding Principles

The discourse state should be:

* Lightweight
* Explicit
* Incrementally updated
* Easily interpretable
* Experimentally validated

---

## Initial Context Representation

Version 1 begins with the smallest useful context.

Initial state includes:

Last Speaker

Previous Speaker

Recent Character Mentions

Candidate Speaker Set

Dialogue Position

Conversation Length

---

## Future Context Components

Future experiments may introduce additional context.

Possible additions include:

* Active character weights
* Dialogue turn history
* Local paragraph context
* Mention recency
* Conversation continuity

No component should be added without experimental evidence.

---

## State Update Policy

Context is updated sequentially while reading the novel.

Each processed dialogue may update the state.

State updates must be deterministic whenever possible.

---

## State Complexity

Increasing state size is treated as an experimental variable.

The objective is identifying the minimum contextual representation required for accurate attribution.

---

## Non-Goals

The discourse state will not attempt to represent:

* Character emotions
* Story themes
* World knowledge
* Plot understanding
* Character intentions

These are outside the scope of the project.
