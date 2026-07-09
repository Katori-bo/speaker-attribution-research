# ADR-007: Reject Sparse Addressee State Representation

**Date**: 2026-07-06  
**Status**: Accepted  
**Experiment**: EXP013  

## Decision

Do not include EXP013 addressee features (`candidate_was_addressed`, `addressee_recency`, `speaker_addressee_transition`) in the frozen model. 

EXP013 remains archived as a negative result.

## Reason

1. **No Measurable Improvement**: The evaluation demonstrated a net negative delta (-4 quotes, -0.0016 accuracy) compared to the EXP012B baseline.
2. **Features Show Negligible Importance**: Permutation importance indicated that all three addressee features were effectively ignored by the model (ranking near the very bottom, with `candidate_was_addressed` showing negative impact).
3. **Representation Sparsity**: The capability extraction successfully validated that Speaker-Addressee signals exist, but coverage was extremely sparse (12%). This resulted in mostly empty historical interaction states.
4. **False Recoveries**: While the model recovered 6 quotes, exactly 0 of them had any active addressee features. The recoveries and regressions were driven purely by ranking noise introduced by adding mostly null columns to the tree ensemble. 

The sparse representation did not successfully capture conversational flow in a way the model could leverage over existing discourse and coreference features.
