# EXP012: Representation Survey

## Research Question
> Which lightweight semantic representations correspond to the reasoning capabilities discovered in EXP010?

## Step 1: Capability Mapping
The failure categories from EXP010 directly imply missing reasoning capabilities:

| Failure Category | Missing Capability |
| :--- | :--- |
| Coreference | Resolve pronouns |
| Alias | Entity normalization |
| Speaker continuity | Dialogue participant tracking |
| Addressee reasoning | Interaction modelling |

## Step 2: Representation Candidates
How should this information be represented (not which algorithm/model to use)?

| Capability | Candidate Representation |
| :--- | :--- |
| Coreference | Deterministic mention chains |
| Alias | Canonical entity mapping |
| Dialogue continuity | Local dialogue graph |
| Addressee reasoning | Speaker–listener graph |

## Step 3: Representation Evaluation

| Representation | Complexity | Explainable | Runtime | External Dependency | Fits Philosophy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Deterministic mention chains | Low | Yes | Fast | None/Minimal | Yes |
| Canonical entity mapping | Medium | Yes | Fast | Alias dictionary | Yes |
| Local dialogue graph | High | Yes | Medium | Graph structure | Moderate |
| Speaker-listener graph | High | Yes | Medium | Graph structure | Moderate |

## Step 4: Selection
**Selected Representation: Deterministic Coreference Representation**

*Rationale:* EXP010 demonstrated that coreference failures dominate the remaining errors. A deterministic mention chain representation directly addresses this capability, is highly explainable, introduces low complexity, and perfectly aligns with the project's philosophy of minimal contextual representation.
