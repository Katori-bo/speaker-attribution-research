# EXP016 Design Document: Discourse State Robustness

> **Note:** The official results, including the interaction effect analysis and comparisons with EXP017 mitigations, are documented in [EXP016_RESULTS.md](file:///home/Aditya/speaker-attribution-research/docs/experiments/EXP016_RESULTS.md).

## Research Question
How robust is lightweight discourse reasoning when discourse state tracking becomes imperfect? Specifically, what happens when discourse state must be maintained using the model's own predictions rather than oracle annotations?

## Hypothesis
Oracle discourse state (teacher forcing) significantly overestimates performance, especially on implicit dialogue where structural features like `candidate_is_previous_speaker` dominate. We expect an autoregressive pipeline to suffer from error propagation (cascading errors) through conversational chains, leading to a measurable drop in implicit quote accuracy.

## Independent Variable
**Discourse Update Strategy:**
1. **Teacher-Forced Mode:** `state.update(gold_speaker)`
2. **One-Step Autoregressive Mode:** `state.update(predicted_speaker)` for previous speaker only, with the rest of history staying gold.
3. **Fully Autoregressive Mode:** `state.update(predicted_speaker)` for the entire evaluation.

## Controlled Variables
- **Same Model:** Frozen HistGBM (EXP014 architecture)
- **Same Features:** No new feature engineering or embeddings
- **Same Candidates:** Frozen candidate generation
- **Same Train/Test Split:** Standard PDNC dataset split
- **Same Hyperparameters:** Frozen from EXP014

## Evaluation Metrics & Analyses
1. **Overall Performance Drop:** Total accuracy comparison across all three tracking modes.
2. **Quote-Type Accuracy:** Degradation breakdown using native PDNC labels (Explicit, Anaphoric, Implicit) and derived subsets (Explicit Named, Explicit Nominal).
3. **Conversation-Length Analysis:** Accuracy binned by conversational depth (Length 1, 2, 3... >10) to measure compounding error.
4. **Recovery Analysis (Cascade Length):** Measuring how often a single misattribution causes subsequent quotes in a chain to also fail, and how many quotes it takes for the system to recover its tracking state.
5. **Confidence Drift Analysis:** Tracking prediction probability following an error to determine if the model confidently corrupts its own state.
