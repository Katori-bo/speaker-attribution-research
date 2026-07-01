# Coding Standards

## Philosophy

Code should be understandable six months from now.

Research code is expected to evolve.

Maintainability is therefore critical.

---

## Code Style

Prefer:

Small functions.

Clear naming.

Descriptive variables.

Minimal nesting.

Consistent formatting.

Avoid:

Premature optimization.

Deep inheritance.

Hidden side effects.

Global mutable state.

---

## Modularity

Separate:

Data processing

Feature extraction

State management

Candidate generation

Models

Evaluation

Utilities

Experiments

No component should perform multiple unrelated responsibilities.

---

## Logging

Every important stage should log:

Current operation.

Inputs.

Outputs.

Timing.

Errors.

Warnings.

---

## Configuration

Avoid hard-coded values.

Store configurable parameters in dedicated configuration files.

Experimental settings should be reproducible.

---

## Reproducibility

Random seeds must be fixed where applicable.

Experiments should be repeatable.

Record versions of:

Datasets

Models

Dependencies

Configurations

---

## Error Handling

Fail loudly.

Produce informative error messages.

Do not silently ignore unexpected conditions.

---

## Comments

Write comments explaining:

Why the code exists.

Do not comment obvious syntax.

Good comment:

"Candidate scores are normalized here to ensure comparability across novels."

Poor comment:

"Increment i by one."

---

## Performance

Optimize only after profiling.

Correctness takes priority over speed.

---

## Final Rule

Every line of code should support one of the project's research objectives.

If it does not, reconsider whether it belongs in the repository.
