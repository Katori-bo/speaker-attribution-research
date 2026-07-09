# Audit 1: Discourse Feature Oracle State

## Objective
Determine whether the discourse features rely on oracle speaker information (i.e. information not available at inference time in a true end-to-end setting).

## Analysis
The core attribution features used in EXP014 heavily depend on the `discourse_state.py`. Upon auditing the dataset generation script (`scripts/generate_dataset_p2.py`), we observed the following state update logic:

```python
if gold_speaker != "Unknown":
    previous_speakers.append(gold_speaker)

# For the next quote, the state is updated using the true gold label:
state.update(
    previous_speakers[-1] if previous_speakers else None, 
    explicit_mentions, 
    candidates
)
```

This populates `state.previous_speaker` with the **true gold label** of the preceding quote. 

### Affected Features
The following features are directly computed using this oracle discourse state:
1. `candidate_is_previous_speaker`: Evaluates to 1.0 if the candidate perfectly matches the gold previous speaker.
2. `conversation_speaker_change`: Uses `state.last_speaker` and `state.previous_speaker`.
3. `chain_recency` and `candidate_in_quote_chain`: Relies on conversational chain continuity built on gold tracking.

## Decision: Conditional Evaluation (Not Leakage)
As established by the project parameters, our research task up to EXP014 has been:
> *"Given the current accurate discourse state, predict the speaker."*

Using gold previous speakers is a **conditional evaluation assumption**, similar to assuming an oracle candidate generator. It is perfectly acceptable and common in NLP pipelines (e.g. coreference resolution assuming oracle entity boundaries) as long as it is explicitly stated.

## Limitations & Next Steps
By conditioning on the gold discourse state, the current ~80.9% plateau and 89% implicit accuracy represent the **upper bound** of what this representation can achieve. 

**EXP014 remains valid under this conditional assumption.** We will freeze EXP014's evaluation methodology. 

In future work (EXP016), we must explore an **autoregressive pipeline** where the discourse state is updated using the model's own predictions rather than the gold labels. This will allow us to measure degradation caused by error propagation.
