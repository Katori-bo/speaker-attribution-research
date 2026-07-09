# EXP013: Speaker-Addressee Reasoning Specification

## Hypothesis
> Maintaining explicit addressee state improves speaker attribution in dialogue exchanges where the next speaker is determined by conversational flow rather than mention proximity.

## Baseline
Frozen: `EXP012B` (`EXP011` + `Coreference` + `Alias Representation`)

## Experimental Variable
Only: **Speaker-Addressee Representation**

(No other variables should be introduced.)

---

## Possible Features

Keep it minimal:

### 1. `candidate_was_addressed`
- Previous quote addressed candidate?
- True / False

### 2. `addressee_recency`
- Turns since candidate was addressed

### 3. `speaker_addressee_transition`
- Previous speaker → candidate interaction exists

*Note: Avoid adding features for emotion, relationship graphs, character importance, or social networks. Those become different experiments.*

---

## Execution Plan (Before Implementation)

We will follow the successful research loop established in EXP012:

### EXP013A.0 Data Source Feasibility
Can we extract addressees reliably?

↓

### EXP013A.1 Representation
Design the data structure to represent the Dialogue Interaction State.

↓

### EXP013A.2 Validation
Ensure the representation can be populated and checked against data.

↓

### EXP013A.3 Features
Implement the minimal features based on the representation.

↓

### EXP013A.4 Integration
Integrate the features into the evaluation pipeline alongside the EXP012B baseline.

↓

### EXP013A.5 Evaluation
Run the pipeline and measure the delta in attribution accuracy.
