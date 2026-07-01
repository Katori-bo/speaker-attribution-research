# Assumptions

## Purpose

This document records every significant assumption made during the project.

Assumptions are not facts. They should be tested rather than accepted indefinitely.

Every assumption should eventually be validated, revised, or rejected based on experimental evidence.

---

## A1 — PDNC Dataset Quality

**Assumption:**
The PDNC dataset contains sufficiently accurate speaker attribution annotations to serve as reliable gold labels for training and evaluation.

**Why it exists:**
The project relies on PDNC as the primary benchmark. If annotations are unreliable, all experimental results are compromised.

**Confidence Level:** Medium

**Supporting Evidence:**
PDNC is used in published speaker attribution research. No systematic quality audit has been performed for this project.

**Planned Validation:**
Manual inspection of a sample of annotations during Phase 0 dataset exploration. Report annotation quality in the dataset exploration notebook.

**Current Status:** Untested

---

## A2 — Candidate Ranking Formulation

**Assumption:**
Framing speaker attribution as candidate ranking (P(candidate | quote, context)) is more appropriate than multi-class classification for literary novels.

**Why it exists:**
Novels have variable character sets, making fixed-label classification impractical.

**Confidence Level:** High

**Supporting Evidence:**
Architectural reasoning documented in 03_CANDIDATE_RANKING.md. Variable label spaces are a known challenge in NLP.

**Planned Validation:**
Phase 3a will include a classification baseline for direct comparison. Hypothesis H3 tests this assumption.

**Current Status:** Untested

---

## A3 — Context Sufficiency

**Assumption:**
A small set of explicit contextual features (speaker memory, active characters, recent mentions) captures most of the information needed for speaker attribution.

**Why it exists:**
This is the project's central research bet — that explicit context can replace implicit LLM reasoning.

**Confidence Level:** Medium

**Supporting Evidence:**
Linguistic intuition and related work suggesting that speaker attribution relies heavily on local patterns. No direct evidence yet.

**Planned Validation:**
Incremental evaluation across Phase 3 sub-phases. Hypothesis H1 and H2 test this assumption.

**Current Status:** Untested

---

## A4 — Speaker Alternation Pattern

**Assumption:**
Literary dialogue frequently follows alternating speaker patterns, making speaker memory a strong signal.

**Why it exists:**
Hypothesis H6 depends on this assumption. If literary dialogue does not follow predictable patterns, speaker memory will be less valuable than expected.

**Confidence Level:** Medium-High

**Supporting Evidence:**
Common in conversational dialogue. Literary texts may deviate, especially in multi-character scenes.

**Planned Validation:**
Phase 1 error analysis should quantify how often speaker alternation holds in PDNC. Phase 3b will measure the actual contribution.

**Current Status:** Untested

---

## A5 — BookNLP as External Baseline

**Assumption:**
BookNLP represents a reasonable strong baseline for literary speaker attribution.

**Why it exists:**
The project needs an external comparison point. BookNLP is the most established system available.

**Confidence Level:** High

**Supporting Evidence:**
BookNLP is widely used in computational literary studies and performs well on similar tasks.

**Planned Validation:**
Phase 2 evaluation on PDNC.

**Current Status:** Untested

---

## A6 — Novel-Level Splitting

**Assumption:**
Splitting the PDNC dataset at the novel level (entire novels in train or test, never split across both) prevents data leakage and produces a fair evaluation.

**Why it exists:**
Character names and writing styles are shared within a novel. Quote-level splitting would leak information.

**Confidence Level:** High

**Supporting Evidence:**
Standard practice in cross-document NLP evaluation.

**Planned Validation:**
Verify during Phase 0 dataset preparation that no novel appears in both train and test splits.

**Current Status:** Untested

---

## A7 — Lightweight Model Capacity

**Assumption:**
A lightweight model (small transformer, gradient boosting, or similar) has sufficient capacity to learn the mapping from contextual features to speaker identity.

**Why it exists:**
The project prioritizes lightweight computation. If model capacity is the bottleneck, the research direction may need revision.

**Confidence Level:** Medium

**Supporting Evidence:**
Feature-based models have been effective for many structured NLP tasks. No direct evidence for literary speaker attribution.

**Planned Validation:**
Phase 3 evaluation. If the model plateaus far below the external baseline, capacity limitations should be investigated.

**Current Status:** Untested

---

## Policy

New assumptions should be added as they are identified.

When an assumption is validated or falsified, update its status and reference the supporting experiment.
