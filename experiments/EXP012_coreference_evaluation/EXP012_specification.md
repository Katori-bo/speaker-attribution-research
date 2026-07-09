# EXP012 Specification: Deterministic Coreference Evaluation

## Research Question
Can reproducible deterministic entity resolution (coreference chains) recover the Coreference failures left by the explicit conversation-state baseline?

## Hypothesis
Providing the attribution model with explicit coreference chains will significantly reduce the proportion of Coreference-related failures by allowing the model to link uninformative pronouns and nominals directly to canonical candidate entities.

## Baseline
- **Model:** Frozen HistGradientBoosting model.
- **Features:** EXP009 baseline representation.

## Experimental System
- **Model:** Same architecture as the baseline.
- **Representation Addition:** Coreference chains (linking mention spans to canonical entity IDs) appended as features for candidate ranking.

## Independent Variable
- The presence of coreference-derived features during candidate scoring.

## Dependent Variables
- Overall attribution accuracy.
- Coreference category recovery rate (percentage of previously failed Coreference quotes resolved correctly).
- Runtime overhead per novel.
- Memory/storage overhead for the extracted representation.

## Dataset
- Standard validation dataset (with PDNC quotation spans to avoid construct-validity threats).

## Evaluation
- Train experimental system on the expanded feature set.
- Evaluate on the validation set.
- Compare predictions directly against the baseline's predictions to identify specific recovered instances.

### Success Criteria

| Metric | Success Criterion |
| :--- | :--- |
| Overall accuracy | Improvement over EXP011 |
| Coreference-category recovery | Statistically meaningful improvement over baseline (target: >15% recovery) |
| Runtime | Acceptable preprocessing overhead (no more than 2x baseline) |
| Memory | Modest increase consistent with lightweight design |
| Explainability | Every resolved feature traceable to a coreference chain |

## Threats to Validity
- **Cascading Errors:** The coreference pipeline may introduce its own errors (e.g., mislinking a pronoun), which the model might over-rely on.
- **Feature Overlap:** The model may struggle to balance the coreference features against existing mention-frequency features.

## Expected Outcomes
- A measurable increase in overall accuracy, driven specifically by resolving quotations where the speaker is referred to via a pronoun in the immediate context.
- Minimal change in Speaker Continuity or Alias Matching failure categories.

## Failure Criteria
The experiment will be considered a negative result (failing to support the hypothesis) if:
1. Overall accuracy does not improve beyond a 0.5% margin.
2. The coreference representation recovers fewer than 15% of the specifically tagged Coreference failures from EXP010.
3. The computational overhead exceeds the project's lightweight constraints without providing commensurate accuracy gains.
