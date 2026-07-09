# Research Hypotheses

## Purpose

This document converts the project's research questions into explicit, testable hypotheses.

Each hypothesis is designed to be validated or falsified through a specific experiment.

These hypotheses are the foundation for the experimental program defined in the phase documents.

---

## H1 — Contextual Sufficiency

**Statement:**
A lightweight model with explicit contextual features (speaker memory, active characters, recent mentions) can achieve at least 80% of the speaker attribution accuracy of large language model based systems on the PDNC dataset.

**Motivation:**
The core research question asks whether explicit context representation can replace implicit LLM reasoning. This hypothesis quantifies that claim.

**Expected Outcome:**
The lightweight student model (Phase 3) achieves competitive accuracy compared to BookNLP (Phase 2) and any LLM baseline (Phase 4).

**Validation Experiment:**
Phase 3 full evaluation compared against Phase 1 and Phase 2 baselines.

**Success Criteria:**
Student model accuracy reaches at least 80% of the best external baseline accuracy on the same PDNC test split.

**Failure Criteria:**
Student model accuracy falls below 60% of the best external baseline, suggesting that implicit reasoning captures information that explicit features cannot.

---

## H2 — Incremental Context Value

**Statement:**
Each additional contextual component (speaker memory, active characters, recent mentions) provides a statistically measurable improvement in attribution accuracy, with diminishing returns as state complexity increases.

**Motivation:**
The project seeks the minimum contextual representation required. This hypothesis tests whether each component contributes meaningfully.

**Expected Outcome:**
Accuracy increases with each added component (Phase 3a → 3b → 3c → 3d), but the marginal gain decreases.

**Validation Experiment:**
Ablation study across Phase 3 sub-phases. Each sub-phase adds exactly one contextual component.

**Success Criteria:**
At least two of the three contextual additions (speaker memory, active characters, recent mentions) produce a measurable accuracy improvement over the previous stage.

**Failure Criteria:**
No contextual addition produces measurable improvement beyond the base candidate ranking model, suggesting that local context alone is sufficient.

---

## H3 — Candidate Ranking Superiority

**Statement:**
Candidate ranking (predicting P(candidate | quote, context)) outperforms direct multi-class classification for speaker attribution in novels with variable character sets.

**Motivation:**
Each novel has a different set of characters, making fixed-label classification impractical. Ranking naturally accommodates variable candidate sets.

**Expected Outcome:**
The ranking formulation produces higher accuracy than a classification baseline trained on the same features.

**Validation Experiment:**
Compare candidate ranking against a classification baseline during Phase 3a.

**Success Criteria:**
Ranking accuracy exceeds classification accuracy on novels with more than 10 characters.

**Failure Criteria:**
Classification accuracy matches or exceeds ranking accuracy, suggesting the ranking formulation adds unnecessary complexity.

---

## H4 — Heuristic Baseline Strength

**Statement:**
Simple rule-based heuristics (nearest mention, dialogue alternation, speech verb patterns) achieve non-trivial speaker attribution accuracy without any machine learning.

**Motivation:**
Establishing a strong heuristic baseline is essential for measuring the marginal value of learned models.

**Expected Outcome:**
Heuristic baselines achieve meaningful accuracy on explicit quotations (those with direct attribution patterns such as "X said").

**Validation Experiment:**
Phase 1 evaluation on PDNC.

**Success Criteria:**
Heuristic baseline achieves above-chance accuracy across all quote types, and above 70% accuracy on explicit quotations.

**Failure Criteria:**
Heuristic baseline performs at or near chance level, suggesting that even explicit attribution patterns are unreliable in literary text.

---

## H5 — Teacher Marginal Value

**Statement:**
LLM teacher supervision (pseudo-labels, soft targets, or reasoning traces) provides measurable but modest improvement over a well-designed lightweight model trained on gold labels alone.

**Motivation:**
Phase 4 investigates whether LLM supervision adds value. This hypothesis frames the expected outcome.

**Expected Outcome:**
Teacher supervision improves accuracy by a small margin (1–5%) over the gold-label student, primarily on difficult implicit quotations.

**Validation Experiment:**
Phase 4 evaluation comparing student-only vs student-with-teacher on the same test split.

**Success Criteria:**
Teacher-supervised student outperforms gold-label student by a statistically significant margin on at least one error category.

**Failure Criteria:**
Teacher supervision provides no measurable improvement, confirming that the lightweight model with explicit context is sufficient.

---

## H6 — Speaker Memory Dominance

**Statement:**
Speaker memory (last speaker, previous speaker) is the single most valuable contextual signal for speaker attribution in conversational dialogue.

**Motivation:**
Literary dialogue often follows alternating speaker patterns. If speaker memory alone explains most conversational attribution, additional context may be unnecessary for this category.

**Expected Outcome:**
The accuracy improvement from adding speaker memory (Phase 3b) is larger than the improvement from any other single contextual addition.

**Validation Experiment:**
Compare the marginal accuracy gain of each contextual component added individually to the base model.

**Success Criteria:**
Speaker memory produces a larger accuracy gain than active characters or recent mentions when added individually.

**Failure Criteria:**
Another contextual signal provides a larger individual gain, suggesting that speaker memory is less important than expected for literary text.

## H7 — Feature Necessity

**Statement:**
A small subset of explicit contextual features explains most of the improvement over the symbolic baseline.

**Motivation:**
We seek to understand not just whether representations work, but *which* ones matter. This tests whether a few strong signals dominate the task.

**Expected Outcome:**
During feature ablation, removing specific feature families (e.g., Conversation features) causes significant accuracy drops, while removing others causes negligible change.

**Validation Experiment:**
Phase 2C Feature Family Ablations on the logistic regression baseline.

**Success Criteria:**
Ablation results demonstrate a highly skewed distribution of feature importance across families.

**Failure Criteria:**
All features contribute equally in small amounts, implying the task relies on a dense, uninterpretable representation rather than discrete discourse signals.

---

## Usage

These hypotheses should be referenced by experiment reports.

Every experiment should state which hypothesis it tests.

Hypotheses may be refined based on experimental evidence, but refinements must be documented with justification.
