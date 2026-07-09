# ADR004: Feature Schema v1 Freeze

## Status
Accepted (Phase 2A)

## Context
In Phase 2, we transitioned from deterministic rule evaluation to candidate ranking using a machine learning model. To ensure scientific rigor and prevent "feature creep" from invalidating experimental results, we must freeze the feature representation before evaluating any models.

## Decision
We freeze Feature Schema v1. The extraction pipeline `src/features/extractor.py` generates the following features:

### Lexical Features
- `lexical_quote_length_chars`
- `lexical_quote_length_tokens`
- `lexical_has_question_mark`
- `lexical_has_exclamation`

### Candidate Features
- `candidate_is_explicit_mention`
- `candidate_is_last_speaker`
- `candidate_is_previous_speaker`
- `candidate_is_recent_mention`

### Discourse Features
- `discourse_dialogue_position`
- `discourse_context_length`

### Conversation Features
- `conversation_turn_index`
- `conversation_length`
- `conversation_speaker_change`

### Symbolic Features
- `symbolic_alternation_rule_fired`
- `symbolic_explicit_rule_fired`

## Consequences
- Any addition or modification of features requires a new documented experiment.
- Downstream models (Logistic Regression, Random Forest, etc.) will be trained exactly on this fixed schema.
- The feature audit (EXP004A) confirmed these features contain no missing values and have non-zero variance.
