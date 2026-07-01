# Research Roadmap

## Purpose

This document describes the experimental decision tree for the project.

It is not an implementation plan. It describes how the outcome of each phase determines the direction of future work.

The roadmap should be read after understanding the research questions (02_RESEARCH_QUESTIONS.md) and the hypotheses (07_RESEARCH_HYPOTHESES.md).

---

## Phase 1 — Heuristic Baseline

**Current Question:**
How much speaker attribution accuracy is achievable without machine learning?

**Possible Outcomes:**

* **High accuracy on explicit quotes (>70%):** Confirms that rule-based methods handle simple cases well. The remaining errors define the challenge for learned models.
* **Low accuracy on explicit quotes (<50%):** Suggests that even explicit attribution patterns are more complex than expected in literary text. May require revisiting dialogue detection.
* **Very low accuracy on implicit quotes:** Expected. Defines the gap that contextual models must close.

**Next Experiment:**
Phase 2 — External Baseline (BookNLP).

**Alternative Path:**
If dialogue detection itself fails significantly, a sub-experiment on dialogue extraction quality may be required before proceeding.

**Decision Criteria:**
Proceed to Phase 2 regardless of outcome. Phase 1 results become the floor for all future comparisons.

---

## Phase 2 — External Baseline (BookNLP)

**Current Question:**
How does an established system perform on PDNC, and what are its failure patterns?

**Possible Outcomes:**

* **BookNLP performs well (>80% accuracy):** Sets a high bar. The student model must demonstrate competitive performance or offer clear advantages in efficiency and interpretability.
* **BookNLP performs moderately (50–80%):** Indicates significant room for improvement through better contextual representation.
* **BookNLP performs poorly (<50%):** Suggests that speaker attribution in literary text is fundamentally harder than in other domains, which strengthens the motivation for the project.

**Next Experiment:**
Phase 3a — Base candidate ranking model.

**Alternative Path:**
If BookNLP fails catastrophically, investigate whether PDNC annotations themselves have quality issues before proceeding.

**Decision Criteria:**
Proceed to Phase 3 regardless. BookNLP results define the external comparison point.

---

## Phase 3a — Base Candidate Ranking

**Current Question:**
Can candidate ranking with local context alone outperform heuristic baselines?

**Possible Outcomes:**

* **Significant improvement over heuristics:** Validates the candidate ranking formulation. Proceed to add contextual components.
* **Marginal improvement:** The ranking model may need better feature representation. Consider whether local context features are sufficient before adding state.
* **No improvement:** Reconsider the feature set or model architecture before adding complexity.

**Next Experiment:**
Phase 3b — Add speaker memory.

**Alternative Path:**
If ranking fails to improve over heuristics, perform error analysis to determine whether the issue is feature quality or model capacity before proceeding.

**Decision Criteria:**
Proceed to Phase 3b if the ranking model demonstrates any measurable improvement over heuristics. If not, conduct error analysis first.

---

## Phase 3b — Speaker Memory

**Current Question:**
Does tracking the last speaker and previous speaker improve attribution accuracy?

**Possible Outcomes:**

* **Clear improvement:** Speaker memory is valuable. Confirms hypothesis H6.
* **Improvement only on conversational dialogue:** Speaker memory helps in conversations but not in isolated quotes. This is an expected and informative result.
* **No improvement:** Speaker memory may be redundant with local context features. Proceed to Phase 3c to test alternative context.

**Next Experiment:**
Phase 3c — Active character representation.

**Alternative Path:**
None. Proceed to Phase 3c regardless.

**Decision Criteria:**
Record the marginal gain. Proceed to Phase 3c.

---

## Phase 3c — Active Character Representation

**Current Question:**
Does tracking which characters are active improve attribution accuracy?

**Possible Outcomes:**

* **Clear improvement:** Active character tracking provides complementary information to speaker memory.
* **Improvement only when speaker memory fails:** Active characters help in non-conversational contexts. This is a valuable finding.
* **No improvement:** Active character tracking may be redundant. Proceed to Phase 3d.

**Next Experiment:**
Phase 3d — Recent mention representation.

**Alternative Path:**
None. Proceed to Phase 3d regardless.

**Decision Criteria:**
Record the marginal gain. Proceed to Phase 3d.

---

## Phase 3d — Recent Mention Representation

**Current Question:**
Does tracking recent character mentions, distances, and frequencies improve attribution accuracy?

**Possible Outcomes:**

* **Clear improvement:** Mention patterns provide information not captured by speaker memory or active characters.
* **Marginal improvement:** Diminishing returns confirm that a saturation point exists.
* **No improvement:** The discourse state has reached sufficient complexity. Additional context is unnecessary.

**Next Experiment:**
Phase 3e — Error analysis.

**Alternative Path:**
None. Phase 3e is mandatory.

**Decision Criteria:**
Record the marginal gain. Proceed to error analysis.

---

## Phase 3e — Error Analysis

**Current Question:**
What categories of errors remain after the full discourse state model?

**Possible Outcomes:**

* **Errors concentrated in implicit quotes:** Suggests that the model lacks sufficient linguistic features for difficult cases. Phase 4 (LLM teacher) may address this.
* **Errors concentrated in long-range attribution:** Suggests that the discourse state window is too narrow. Consider extending context before moving to Phase 4.
* **Errors distributed uniformly:** Suggests a fundamental model capacity limitation rather than a missing feature.

**Next Experiment:**
Phase 4a — Teacher evaluation (if justified) or Phase 5 — Final evaluation.

**Alternative Path:**
If error analysis reveals a specific missing feature, a targeted Phase 3f experiment may be warranted before proceeding to Phase 4.

**Decision Criteria:**
If error analysis identifies categories that an LLM might address, proceed to Phase 4. Otherwise, proceed directly to Phase 5.

---

## Phase 4 — LLM Extension (Optional)

**Current Question:**
Does LLM teacher supervision provide measurable additional value?

**Possible Outcomes:**

* **Measurable improvement:** Quantify exactly what the teacher contributes and at what cost.
* **No improvement:** Document this as a positive finding — the lightweight model is sufficient.

**Next Experiment:**
Phase 5 — Final evaluation.

**Decision Criteria:**
Proceed to Phase 5 regardless of outcome.

---

## Phase 5 — Final Evaluation

**Current Question:**
Which contextual representations contributed most to accurate speaker attribution?

**Deliverable:**
Complete scientific evaluation answering all research questions.

This phase produces conclusions, not new experiments.

---

## Summary

The roadmap follows a strict evidence-driven progression:

```
Heuristic Baseline
        ↓
External Baseline
        ↓
Base Ranking Model
        ↓
+ Speaker Memory
        ↓
+ Active Characters
        ↓
+ Recent Mentions
        ↓
Error Analysis → Decision Point
        ↓                  ↓
LLM Extension      Final Evaluation
        ↓
Final Evaluation
```

Every transition between phases depends on the outcome of the previous phase.

No phase should be skipped without documented justification.
