# EXP008: Nonlinear Sanity Check Report

Does a lightweight nonlinear learner extract meaningful additional signal from the explicit representation?

## 1. Performance Comparison

| Model                       |   Accuracy |   ROC-AUC |   PR-AUC |   LogLoss |      ECE |   Train_Time_sec |   Eval_Time_sec |
|:----------------------------|-----------:|----------:|---------:|----------:|---------:|-----------------:|----------------:|
| Logistic Regression (Top 3) |   0.841017 |  0.91502  | 0.756769 |  0.343772 | 0.158736 |        0.0887246 |      0.00236535 |
| HistGBM (All Features)      |   0.852219 |  0.943146 | 0.866691 |  0.290749 | 0.128586 |        2.21917   |      0.00993228 |

- **Δ Accuracy:** 1.120%
- **Δ ROC-AUC:** 0.0281
- **Δ LogLoss:** 0.0530

## 2. Feature Importance (HistGBM)

Permutation importance (scoring=roc_auc) on a holdout sample:

| Feature                       |   Importance_Mean |   Importance_Std |
|:------------------------------|------------------:|-----------------:|
| candidate_is_previous_speaker |       0.14088     |      0.00149084  |
| candidate_is_recent_mention   |       0.125173    |      0.00680486  |
| candidate_is_explicit_mention |       0.0482093   |      0.000820505 |
| candidate_is_last_speaker     |       0.0170269   |      0.00129197  |
| discourse_context_length      |       0.0103054   |      0.000667677 |
| conversation_speaker_change   |       0.00469205  |      0.000478503 |
| lexical_quote_length_tokens   |       0.00162943  |      0.000801964 |
| lexical_quote_length_chars    |       0.000614858 |      0.000304315 |
| lexical_has_exclamation       |       0.00019923  |      0.000113618 |
| lexical_has_question_mark     |       0.000197857 |      6.46402e-05 |
| discourse_dialogue_position   |       0           |      0           |
| conversation_turn_index       |       0           |      0           |
| conversation_length           |      -0.000235393 |      0.000402744 |

