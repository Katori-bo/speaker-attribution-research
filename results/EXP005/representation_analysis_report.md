# EXP005: Representation Analysis Report

## 1. Coefficient Analysis
Features sorted by absolute standardized coefficient magnitude. This highlights their independent importance.

| Feature | Coefficient | Std Error | 95% CI | p-value | Odds Ratio |
|---------|-------------|-----------|--------|---------|------------|
| candidate_is_recent_mention | -1.8505 | 0.0142 | [-1.8783, -1.8227] | 0.0000e+00 | 0.1572 |
| candidate_is_previous_speaker | 1.4200 | 0.0089 | [1.4026, 1.4375] | 0.0000e+00 | 4.1373 |
| candidate_is_explicit_mention | 0.6782 | 0.0142 | [0.6505, 0.7060] | 0.0000e+00 | 1.9704 |
| lexical_quote_length_chars | -0.5987 | 0.1443 | [-0.8816, -0.3158] | 3.3516e-05 | 0.5495 |
| lexical_quote_length_tokens | 0.5726 | 0.1436 | [0.2913, 0.8540] | 6.6353e-05 | 1.7729 |
| candidate_is_last_speaker | 0.5092 | 0.0110 | [0.4876, 0.5308] | 0.0000e+00 | 1.6640 |
| discourse_context_length | -0.1949 | 0.0109 | [-0.2163, -0.1734] | 4.0425e-71 | 0.8229 |
| lexical_has_exclamation | -0.0350 | 0.0096 | [-0.0539, -0.0162] | 2.7366e-04 | 0.9656 |
| lexical_has_question_mark | -0.0327 | 0.0093 | [-0.0509, -0.0145] | 4.1812e-04 | 0.9678 |
| conversation_speaker_change | -0.0236 | 0.0090 | [-0.0411, -0.0060] | 8.5814e-03 | 0.9767 |
| conversation_length | -0.0075 | nan | [nan, nan] | nan | 0.9925 |
| discourse_dialogue_position | -0.0075 | nan | [nan, nan] | nan | 0.9925 |
| conversation_turn_index | -0.0075 | nan | [nan, nan] | nan | 0.9925 |

## 2. Representation Sufficiency (Forward Selection)
How much representation is actually necessary? We start with the best feature and incrementally add the next best.

![Representation Sufficiency Curve](/home/Aditya/speaker-attribution-research/results/EXP005/sufficiency_curve.png)

| Features Used | Latest Feature Added | Ranking Accuracy |
|---------------|----------------------|------------------|
| 1 | candidate_is_recent_mention | 55.32% |
| 2 | candidate_is_previous_speaker | 78.24% |
| 3 | candidate_is_explicit_mention | 84.10% |
| 4 | lexical_quote_length_chars | 84.10% |
| 5 | lexical_quote_length_tokens | 84.10% |
| 6 | candidate_is_last_speaker | 84.23% |
| 7 | discourse_context_length | 84.23% |
| 8 | lexical_has_exclamation | 84.23% |
| 9 | lexical_has_question_mark | 84.23% |
| 10 | conversation_speaker_change | 84.23% |
| 11 | conversation_length | 84.23% |
| 12 | discourse_dialogue_position | 84.23% |
| 13 | conversation_turn_index | 84.23% |

## 3. Single-Feature Ablation
Baseline Accuracy (All features): 84.23%

| Ablated Feature | New Accuracy | Absolute Drop |
|-----------------|--------------|---------------|
| candidate_is_recent_mention | 76.82% | 7.41% |
| candidate_is_previous_speaker | 71.05% | 13.18% |
| candidate_is_explicit_mention | 78.16% | 6.07% |
| lexical_quote_length_chars | 84.23% | 0.00% |
| lexical_quote_length_tokens | 84.23% | 0.00% |

## 4. Probability Calibration
Since this is a ranking task, probability calibration is critical.

![Reliability Diagram](/home/Aditya/speaker-attribution-research/results/EXP005/reliability_diagram.png)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9330 |
| PR-AUC | 0.8421 |
| Brier Score | 0.0982 |
| Expected Calibration Error (ECE) | 0.1451 |
