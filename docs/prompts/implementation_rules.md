# Implementation Rules

## Objective

Produce clean, reproducible implementations that directly support the research.

---

## General Principles

* Simplicity before complexity.
* Readability before cleverness.
* Modularity before optimization.
* Correctness before performance.

---

## Component Rules

Every component must have:

Purpose

Inputs

Outputs

Dependencies

Acceptance Criteria

Failure Conditions

---

## Before Writing Code

Confirm:

* Which phase is currently active.
* Which experiment the implementation supports.
* Which research question it contributes to.

---

## Feature Additions

Every new feature must include:

Why it exists.

What problem it solves.

Expected improvement.

How it will be evaluated.

If these cannot be answered, do not implement the feature.

---

## Refactoring Rules

Refactoring is allowed only if it:

Improves readability.

Improves modularity.

Reduces duplication.

Does not change experimental behaviour.

---

## Documentation

Every public module should explain:

Why it exists.

How it fits into the research.

Which experiment depends on it.

---

## Dependencies

Prefer:

Small

Stable

Well-maintained

Libraries.

Avoid unnecessary frameworks.

---

## Testing

Every implemented component should have:

Unit tests when practical.

Validation examples.

Failure case handling.

Meaningful logging.
