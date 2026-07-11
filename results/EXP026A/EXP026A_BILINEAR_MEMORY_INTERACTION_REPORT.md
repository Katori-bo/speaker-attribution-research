# EXP026A Bilinear Candidate-History Compatibility

Regardless of statistical significance, EXP026A is the final scorer-integration experiment for the current feature-only GRU architecture.

## Design

- Variant A: `nomemory_candidate_only`, candidate encoder plus `f(c)`, no GRU.
- Variant B: `gru_concatenative_current`, current EXP025 GRU scorer.
- Variant C: `gru_bilinear_interaction`, `s(c,h)=f(c)+c^T W h`.
- No auxiliary loss, no candidate-feature dropout, no alternate fusion scorer.

## Parameter Counts

| Model                      |   Trainable_Params |   Scorer_Only_Params |   Bilinear_W_Params |   Seed |
|:---------------------------|-------------------:|---------------------:|--------------------:|-------:|
| nomemory_candidate_only    |               7297 |                 2113 |                   0 |      1 |
| gru_concatenative_current  |              34337 |                 4193 |                   0 |      1 |
| gru_bilinear_interaction   |              36353 |                 6209 |                4096 |      1 |
| nomemory_parameter_matched |              36283 |                31099 |                   0 |      1 |
| nomemory_candidate_only    |               7297 |                 2113 |                   0 |      2 |
| gru_concatenative_current  |              34337 |                 4193 |                   0 |      2 |
| gru_bilinear_interaction   |              36353 |                 6209 |                4096 |      2 |
| nomemory_parameter_matched |              36283 |                31099 |                   0 |      2 |
| nomemory_candidate_only    |               7297 |                 2113 |                   0 |      3 |
| gru_concatenative_current  |              34337 |                 4193 |                   0 |      3 |
| gru_bilinear_interaction   |              36353 |                 6209 |                4096 |      3 |
| nomemory_parameter_matched |              36283 |                31099 |                   0 |      3 |
| nomemory_candidate_only    |               7297 |                 2113 |                   0 |      4 |
| gru_concatenative_current  |              34337 |                 4193 |                   0 |      4 |
| gru_bilinear_interaction   |              36353 |                 6209 |                4096 |      4 |
| nomemory_parameter_matched |              36283 |                31099 |                   0 |      4 |
| nomemory_candidate_only    |               7297 |                 2113 |                   0 |      5 |
| gru_concatenative_current  |              34337 |                 4193 |                   0 |      5 |
| gru_bilinear_interaction   |              36353 |                 6209 |                4096 |      5 |
| nomemory_parameter_matched |              36283 |                31099 |                   0 |      5 |


Bilinear interaction adds `64 x 64 = 4096` parameters. A parameter-matched no-memory control is required only if Variant C exceeds Variant A by more than 10% trainable parameters.

## Seed Summary

| Model                      |   Accuracy_Mean |   Accuracy_Std |   Accuracy_Min |   Accuracy_Max |   Implicit_Mean |   Anaphoric_Mean |   MRR_Mean |
|:---------------------------|----------------:|---------------:|---------------:|---------------:|----------------:|-----------------:|-----------:|
| gru_bilinear_interaction   |        0.696184 |     0.0155176  |       0.680048 |       0.719396 |        0.610223 |         0.583013 |   0.801093 |
| gru_concatenative_current  |        0.691176 |     0.00555729 |       0.686407 |       0.700715 |        0.581389 |         0.593269 |   0.796451 |
| nomemory_candidate_only    |        0.707631 |     0.0238252  |       0.680048 |       0.728537 |        0.626999 |         0.602885 |   0.806572 |
| nomemory_parameter_matched |        0.704928 |     0.0108324  |       0.689189 |       0.716614 |        0.619659 |         0.600962 |   0.806489 |

## Ablation Summary

| Model                                                    |   Accuracy_Mean |   Implicit_Mean |   Anaphoric_Mean |   MRR_Mean |
|:---------------------------------------------------------|----------------:|----------------:|-----------------:|-----------:|
| gru_bilinear_interaction_shuffled_update                 |        0.692369 |        0.610747 |         0.575641 |   0.797959 |
| gru_bilinear_interaction_teacher_forced_eval_diagnostic  |        0.712639 |        0.651376 |         0.59359  |   0.812153 |
| gru_bilinear_interaction_zero_state                      |        0.654372 |        0.556488 |         0.553846 |   0.770873 |
| gru_concatenative_current_shuffled_update                |        0.692289 |        0.585845 |         0.592308 |   0.797083 |
| gru_concatenative_current_teacher_forced_eval_diagnostic |        0.691892 |        0.58401  |         0.592628 |   0.796875 |
| gru_concatenative_current_zero_state                     |        0.694833 |        0.596068 |         0.590385 |   0.798664 |

## Interaction Magnitude

|   Mean_Abs_Interaction |   Median_Abs_Interaction |   P95_Abs_Interaction |   Seed | Model                    |
|-----------------------:|-------------------------:|----------------------:|-------:|:-------------------------|
|                1.90766 |                  1.63866 |               4.99535 |      1 | gru_bilinear_interaction |
|                1.45485 |                  1.09481 |               3.86518 |      2 | gru_bilinear_interaction |
|                1.73647 |                  1.4764  |               4.3856  |      3 | gru_bilinear_interaction |
|                2.415   |                  2.34313 |               5.0628  |      4 | gru_bilinear_interaction |
|                1.65206 |                  1.49429 |               4.21322 |      5 | gru_bilinear_interaction |

## Probe Controls

|   Seed | Probe                                       |   Previous_Speaker_PR_AUC |
|-------:|:--------------------------------------------|--------------------------:|
|      1 | candidate_only                              |                  0.414503 |
|      1 | candidate_plus_aligned_hidden               |                  0.402344 |
|      1 | candidate_plus_within_novel_shuffled_hidden |                  0.397355 |
|      2 | candidate_only                              |                  0.402338 |
|      2 | candidate_plus_aligned_hidden               |                  0.383307 |
|      2 | candidate_plus_within_novel_shuffled_hidden |                  0.387364 |
|      3 | candidate_only                              |                  0.409754 |
|      3 | candidate_plus_aligned_hidden               |                  0.400122 |
|      3 | candidate_plus_within_novel_shuffled_hidden |                  0.395778 |
|      4 | candidate_only                              |                  0.438837 |
|      4 | candidate_plus_aligned_hidden               |                  0.419448 |
|      4 | candidate_plus_within_novel_shuffled_hidden |                  0.435505 |
|      5 | candidate_only                              |                  0.4226   |
|      5 | candidate_plus_aligned_hidden               |                  0.410292 |
|      5 | candidate_plus_within_novel_shuffled_hidden |                  0.402861 |

## Acceptance Rules

- Predictive: Variant C must exceed the five-seed no-memory baseline 71.97% by at least +0.5 pp, with positive paired seed delta and wins on most seeds.
- Causal: normal - zero_state must be at least +0.5 pp, normal must exceed shuffled_update, and teacher-forced must not be materially or consistently worse than normal.
- Probe success alone cannot override failed attribution results.
- If `c^T W h` is near zero almost everywhere, do not interpret any accuracy gain as memory interaction.

## Outcome Rules

- Accuracy and memory dependence both succeed: promote Variant C, then interpret what memory learned.
- Accuracy improves but zero-state does not hurt: treat the gain as static ranking, not recurrent memory.
- Memory sensitivity increases but accuracy does not: EXP026B auxiliary supervision is justified.
- Neither improves: retire the current feature-only GRU branch; do not run EXP026B, EXP026C, deeper fusion, attention, or another scorer-integration variant.