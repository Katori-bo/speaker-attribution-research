# Candidate Ranking

## Purpose

The system predicts the most probable speaker from a dynamically generated candidate set.

Candidate ranking replaces traditional multi-class classification.

---

## Motivation

Each novel contains different characters.

Therefore the label space changes between books.

Ranking naturally handles variable candidate sets.

---

## Prediction Objective

Given:

Dialogue

Candidate Set

Context Representation

Predict:

Probability of each candidate being the speaker.

The highest-ranked candidate becomes the prediction.

---

## Advantages

Candidate ranking:

* Supports variable candidate sets
* Produces confidence scores
* Enables uncertainty analysis
* Allows future calibration methods
* Integrates naturally with contextual information

---

## Candidate Generation

Candidate generation is intentionally separated from ranking.

Generation focuses on recall.

Ranking focuses on precision.

---

## Evaluation

Ranking quality will be evaluated using speaker attribution accuracy.

Future experiments may evaluate confidence calibration and ranking quality.

---

## Future Extensions

Potential future investigations include:

* Pairwise ranking
* Listwise ranking
* Neural ranking models
* Gradient boosting rankers
* Sequential ranking

These are research questions rather than fixed architectural decisions.
