# Phase 4C: Representation Justification Audit

## Goal
Before implementing any of the representations surveyed in Phase 4B, they must be audited against the existing frozen baseline to ensure they introduce genuinely novel contextual information and are methodologically feasible.

## Audit 1: Coreference Chains

1. **Is it already present in EXP009/EXP011?**
   No. The baseline tracks unlinked character mentions (frequencies and distances) but does not resolve pronouns or possessives to canonical entities.
2. **Does this introduce new information or merely reorganize existing information?**
   It introduces entirely new semantic information: explicit semantic links between anaphoric text spans and their referent entities.
3. **Can it be extracted deterministically?**
   Given fixed inputs and model versions, a preprocessing pipeline produces reproducible coreference annotations.
4. **Does it require external models?**
   Yes. It requires an external preprocessing step (e.g., BookNLP).
5. **Is there a simpler alternative?**
   Simple string matching is insufficient for pronouns ("he", "she") and complex nominals ("the doctor").
6. **How will success be measured?**
   Accuracy recovery specifically within the *Coreference* failure category identified in EXP010.

## Audit 2: Canonical Alias Dictionary

1. **Is it already present in EXP009/EXP011?**
   Partially. EXP009 relies on exact string matches for candidate generation.
2. **Does this introduce new information or merely reorganize existing information?**
   It introduces new relational information linking disparate noun phrases (e.g., "Mr. Smith" and "John") to the same underlying entity ID.
3. **Can it be extracted deterministically?**
   Yes, via reproducible character alias clustering.
4. **Does it require external models?**
   No, it can be constructed heuristically or from standard pipeline metadata.
5. **Is there a simpler alternative?**
   No. This is the simplest possible representation for alias matching.
6. **How will success be measured?**
   Accuracy recovery specifically within the *Alias Matching* failure category.

## Audit 3: Conversation Graph

1. **Is it already present in EXP009/EXP011?**
   EXP011 attempted to model conversational continuity linearly (Participant Stack).
2. **Does this introduce new information or merely reorganize existing information?**
   A conversation graph is only justified if it captures interaction information unavailable to the explicit conversation-state features already evaluated. If it simply re-encodes the participant stack as nodes, it provides no new information.
3. **Can it be extracted deterministically?**
   No. It often requires implicit inference (knowing who is answering whom).
4. **Feasibility Threat:**
   Constructing an accurate dialogue graph deterministically from raw text is an open NLP problem and introduces significant cascading error risk.

## Phase Gate Decision

To ensure an objective and evidence-driven transition to Phase 4 experimentation, candidate representations are scored against the following criteria:

| Criterion | Weight |
| :--- | :--- |
| Frequency in EXP010 | 40% |
| Novelty | 20% |
| Simplicity | 20% |
| Expected impact | 20% |

### Representation Scoring

| Representation | Score | Rationale |
| :--- | :---: | :--- |
| Coreference Chains | 91 | High frequency (~20% of errors), completely novel, high impact, manageable complexity. |
| Alias Dictionary | 82 | Extremely simple, reproducible, no external dependency, but lower overall frequency than coreference. |
| Conversation Graph | 58 | Novelty depends heavily on implementation; high complexity and cascading error risk. |
| Local Dialogue Graph | 44 | Highest complexity and extraction difficulty; targets a smaller, ambiguous error category. |

**Selected Capability for EXP012: Coreference Resolution**
**Selected Representation: Coreference Chains**

*Note:* Because the Canonical Alias Dictionary is extremely cheap and completely reproducible (Score: 82), it may be evaluated as an intermediate step (e.g., EXP012A Alias Dictionary -> EXP012B Coreference) depending on resource availability. However, Coreference remains the primary target for recovering the dominant residual errors.
