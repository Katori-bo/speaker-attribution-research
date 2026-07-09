# Phase 3 Conclusions

## 3.1 Scientific Findings
- EXP010 successfully identified the dominant failure mechanisms.
- Feature Audit confirmed novelty.
- Feasibility Audit eliminated construct-validity concerns.
- EXP011 demonstrated that richer explicit conversation-state representations provide only marginal additional predictive information.

## 3.2 Methodological Findings
Construct validity of contextual representations must be verified before evaluation. The Feature Audit and Representation Feasibility Audit prevented the project from drawing incorrect conclusions from an unfaithful implementation.

## 3.3 Why explicit state failed
The evaluated explicit discourse-state variables encode structural properties of dialogue but do not provide the entity-level or relational information required to resolve the dominant residual failure categories identified in EXP010.

## 3.4 Remaining failures
Based on EXP010 and the subsequent EXP011 findings, the dominant remaining failure categories are:
- **Coreference:** Failure to resolve pronouns and nominal references.
- **Alias Matching:** Failure to normalize alternate names and titles.
- **Speaker Continuity:** Conversation continuity not explained by explicit conversation-state memory.
- **Speaker–Addressee Reasoning:** Failure to infer conversational roles.

## 3.5 New hypotheses
Given the failure of explicit discourse state to address these issues, the project shifts towards a new hypothesis:

**Minimal, targeted semantic representations (such as deterministic coreference chains or local dialogue graphs) are expected to recover the dominant failure categories more effectively and efficiently than expanded explicit discourse state.**

This leads directly into Phase 4, which will investigate and implement these lightweight semantic representations.

## 3.6 Phase 3 Scientific Contribution
Phase 3 demonstrated that identifying failure categories is not sufficient to justify an architectural solution. Through a sequence of Feature Audits, Representation Feasibility Audits, faithful implementation, and controlled evaluation, the project showed that the dominant residual error category (Speaker Continuity) is not adequately addressed by richer explicit conversation-state representations. This negative result narrows the search space for future work and motivates investigation of lightweight semantic reasoning mechanisms rather than increasingly complex discourse-state tracking.
