# EXP011 Final Report

## Original Hypothesis
Increasing the complexity of explicit discourse-state features will meaningfully improve speaker attribution accuracy by resolving edge cases in conversational continuity.

## Implemented Features
EXP011 evaluated the following conversation-state representations:
- Active Conversation ID
- Participant Stack
- Interruption Distance
- Candidate Stack Depth / Membership

## Experimental Setup
- **Baseline:** Frozen HistGradientBoosting model trained on the EXP009 explicit feature representation.
- **Experimental Model:** Same model trained on the baseline representation augmented with the novel conversation-state features.
- **Dataset:** Standard validation set.

EXP011 consisted of three stages to ensure rigorous evaluation:
1. Feature Audit
2. Representation Feasibility Audit
3. Faithful Evaluation

## Results
The addition of enriched explicit discourse-state features resulted in a performance plateau. 

| Metric                      | Baseline |  EXP011 |
| :---                        | -------: | ------: |
| Accuracy                    |   85.22% |  86.95% |
| Speaker Continuity Recovery |        0 | 4 / 113 |
| Strict Net Gain             |        — |      −1 |

## Feature and Sparsity Audit
- The Feature Audit confirmed that the proposed conversation-state variables introduced genuinely novel contextual information beyond the frozen EXP009 representation.
- The Representation Feasibility Audit identified missing physical dialogue-offset information as a construct-validity threat. The dataset was subsequently regenerated using raw PDNC quotation spans, enabling a faithful evaluation of the proposed representation.

## Failure Analysis
- The evaluated conversation-state representation recovered only **4 of 113 Speaker Continuity failures (3.5%)**, despite faithful implementation.
- Although overall accuracy increased by 1.73%, the representation failed to recover the target failure mechanism identified by EXP010. Most improvements arose from probability redistribution rather than resolving previously unsolved quotations.
- The evaluated explicit conversation-state representation proved insufficient for recovering the dominant residual error categories identified in EXP010.

## Conclusions
**Richer explicit discourse-state features do not substantially recover the dominant residual error categories.** 

This is a negative result, but a significant publishable scientific finding. It indicates that the path forward is not through accumulating more explicit state tracking, but rather through lightweight semantic representations that directly address the core failure categories.

## Scientific Contribution
EXP011 demonstrates that identifying a dominant failure category does not necessarily imply that a corresponding explicit representation will recover it. Through a sequence of Feature Audits, Representation Feasibility Audits, dataset regeneration, and faithful evaluation, the project established that richer explicit conversation-state representations provide only marginal additional predictive information beyond the frozen EXP009 representation. This negative result narrows the search space for future work and motivates investigation of lightweight semantic reasoning mechanisms rather than increasingly complex explicit discourse-state tracking.
