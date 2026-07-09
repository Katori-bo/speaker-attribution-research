# EXP006: Contextual Redundancy Analysis Report

## Experiment A: Correlation Structure
Full matrix saved to `correlation_matrix.csv`. High correlations indicate redundancy prior to modeling.

## Experiment B: Conditional Importance
If the Top 3 are present, does adding a 4th feature improve probabilities or accuracy?

| Added_Feature               |   Accuracy |   ROC-AUC |   LogLoss |      ECE |   d_Accuracy |    d_ROC-AUC |     d_LogLoss |         d_ECE |
|:----------------------------|-----------:|----------:|----------:|---------:|-------------:|-------------:|--------------:|--------------:|
| None (Baseline)             |   0.841017 |  0.91502  |  0.343772 | 0.158736 | nan          | nan          | nan           | nan           |
| candidate_is_last_speaker   |   0.842309 |  0.925143 |  0.329492 | 0.148602 |   0.00129255 |   0.0101232  |  -0.0142802   |  -0.0101335   |
| conversation_length         |   0.841017 |  0.916831 |  0.345586 | 0.160085 |   0          |   0.00181133 |   0.00181406  |   0.00134971  |
| conversation_speaker_change |   0.841017 |  0.917878 |  0.342168 | 0.157165 |   0          |   0.00285846 |  -0.00160334  |  -0.00157083  |
| conversation_turn_index     |   0.841017 |  0.916831 |  0.345586 | 0.160085 |   0          |   0.00181133 |   0.00181406  |   0.00134971  |
| discourse_context_length    |   0.841017 |  0.916926 |  0.340177 | 0.156524 |   0          |   0.00190642 |  -0.00359452  |  -0.00221138  |
| discourse_dialogue_position |   0.841017 |  0.916831 |  0.345586 | 0.160085 |   0          |   0.00181133 |   0.00181406  |   0.00134971  |
| lexical_has_exclamation     |   0.841017 |  0.918278 |  0.342049 | 0.157636 |   0          |   0.00325828 |  -0.00172321  |  -0.00109938  |
| lexical_has_question_mark   |   0.841017 |  0.917268 |  0.344404 | 0.159228 |   0          |   0.0022483  |   0.000632443 |   0.000492271 |
| lexical_quote_length_chars  |   0.841017 |  0.913778 |  0.343285 | 0.158314 |   0          |  -0.00124183 |  -0.000487054 |  -0.000421913 |
| lexical_quote_length_tokens |   0.841017 |  0.913378 |  0.343363 | 0.158371 |   0          |  -0.00164215 |  -0.000408744 |  -0.000364164 |

## Experiment C: Residual Error Taxonomy
What kind of errors remain when only using the Top 3 features?

| Error Type                            |   Count |
|:--------------------------------------|--------:|
| Gold has no explicit signals          |     224 |
| Unknown                               |      64 |
| Confused Mention for Previous Speaker |      63 |
| Long Context Narration                |      18 |

## Experiment D: Interaction Testing
Do hypothesis-driven feature interactions provide complementary signal?

| Interaction                                              |   Accuracy |   d_Accuracy |
|:---------------------------------------------------------|-----------:|-------------:|
| candidate_is_previous_speaker_X_discourse_context_length |   0.813012 | -0.0280052   |
| candidate_is_previous_speaker_X_conversation_length      |   0.840155 | -0.000861698 |
| candidate_is_explicit_mention_X_discourse_context_length |   0.841448 |  0.000430849 |

## Experiment E: Stability Analysis
Does feature selection converge to the Top-3 set regardless of search strategy (3-fold CV)?

- **Forward Selection Top 3:** ['candidate_is_explicit_mention', 'candidate_is_previous_speaker', 'candidate_is_recent_mention']
- **Backward Elimination Top 3:** ['candidate_is_explicit_mention', 'candidate_is_previous_speaker', 'candidate_is_recent_mention']
