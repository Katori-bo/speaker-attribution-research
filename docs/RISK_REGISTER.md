# Risk Register

## Purpose

This document records research risks that could affect the project's ability to answer its research questions.

Focus is on research and experimental risks rather than software engineering risks.

---

## R1 — Dataset Annotation Quality

**Description:**
PDNC annotations may contain errors or inconsistencies that distort training and evaluation.

**Probability:** Medium

**Impact:** High

**Detection Method:**
Manual inspection during Phase 0 dataset exploration. Inter-annotator agreement analysis if multiple annotations exist.

**Mitigation Strategy:**
Sample-based quality audit during Phase 0. Document known annotation issues. Report metrics with and without suspected noisy examples.

**Contingency Plan:**
If annotation quality is unacceptable, investigate alternative datasets or perform manual correction of a smaller subset.

**Current Status:** Unmitigated — requires Phase 0 dataset exploration.

---

## R2 — Candidate Generation Recall Failure

**Description:**
The candidate generation component may fail to include the correct speaker in the candidate set, creating an upper bound on system accuracy.

**Probability:** Medium

**Impact:** Critical (correct speaker cannot be predicted if not in candidate set)

**Detection Method:**
Measure candidate recall during Phase 1 and Phase 3. Track the percentage of quotes where the gold speaker appears in the generated candidate set.

**Mitigation Strategy:**
Prioritize recall over precision in candidate generation. Use generous candidate windows. Monitor recall as a primary diagnostic metric.

**Contingency Plan:**
If recall is insufficient, expand candidate generation strategy before investing in ranking model improvements.

**Current Status:** Unmitigated — requires Phase 1 implementation.

---

## R3 — Insufficient Heuristic Baseline

**Description:**
If the heuristic baseline is too weak, it becomes an uninformative comparison point and does not help measure the marginal value of learned models.

**Probability:** Low

**Impact:** Medium

**Detection Method:**
Phase 1 evaluation. If heuristic accuracy is near zero on most quote types, the baseline is uninformative.

**Mitigation Strategy:**
Ensure heuristic rules cover multiple attribution patterns (speech verbs, nearest mention, dialogue alternation). Evaluate per quote type.

**Contingency Plan:**
Strengthen heuristic rules iteratively. Use BookNLP (Phase 2) as the primary baseline if heuristics remain too weak.

**Current Status:** Unmitigated — requires Phase 1 implementation.

---

## R4 — Contextual Feature Saturation

**Description:**
Adding more contextual features may produce diminishing returns earlier than expected, leaving a large accuracy gap between the lightweight model and the external baseline.

**Probability:** Medium

**Impact:** Medium

**Detection Method:**
Track marginal accuracy gain at each Phase 3 sub-phase. If gains plateau before reaching competitive accuracy, saturation has occurred.

**Mitigation Strategy:**
Error analysis at Phase 3e to identify whether remaining errors require different types of context or more model capacity.

**Contingency Plan:**
Consider increasing model capacity or exploring feature interactions before moving to LLM teacher supervision.

**Current Status:** Unmitigated — requires Phase 3 experiments.

---

## R5 — Reproducibility Failure

**Description:**
Experiments may produce inconsistent results across runs due to randomness, hardware differences, or environment drift.

**Probability:** Low-Medium

**Impact:** High

**Detection Method:**
Run key experiments with multiple random seeds. Compare results across runs.

**Mitigation Strategy:**
Fix random seeds. Record all package versions. Use deterministic operations where possible. Document hardware and environment.

**Contingency Plan:**
Report confidence intervals rather than point estimates. Increase the number of evaluation runs for key results.

**Current Status:** Partially mitigated — environment setup is part of Phase 0.

---

## R6 — BookNLP Integration Difficulty

**Description:**
BookNLP may be difficult to install, may produce output in an incompatible format, or may not run reliably on the PDNC dataset.

**Probability:** Medium

**Impact:** Low (delays Phase 2 but does not block the core research)

**Detection Method:**
Installation and initial test run during Phase 2.

**Mitigation Strategy:**
Allocate time for output format conversion. Document installation steps. Test on a small subset before full evaluation.

**Contingency Plan:**
If BookNLP cannot be integrated, use published benchmark results from the literature as the external comparison point.

**Current Status:** Unmitigated — requires Phase 2.

---

## R7 — Scope Creep

**Description:**
The project may gradually expand beyond its defined scope (speaker attribution) into general narrative understanding, emotion detection, or other tangential tasks.

**Probability:** Medium

**Impact:** High

**Detection Method:**
Review every proposed addition against the Research Constitution scope definition. Monitor for features that do not directly support speaker attribution.

**Mitigation Strategy:**
Enforce the Research Constitution's scope boundaries. Require every new component to answer the four mandatory questions. Follow the Research Guardrails.

**Contingency Plan:**
If scope creep is detected, halt new development and review all pending work against the constitution.

**Current Status:** Mitigated by documentation — requires ongoing vigilance.

---

## Policy

New risks should be added as they are identified.

When a risk materializes or is resolved, update its status and reference the relevant experiment or decision.
