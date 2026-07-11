# EXP025C GRU Memory Utilization Audit

> EXP025C retrains the EXP025 GRU configuration because original checkpoints were unavailable. Therefore, EXP025C diagnoses the same architecture and training protocol, not the exact original trained parameter instances.

## Diagnostic 1: Hidden-State Movement
|   seed |   mean_hidden_norm |   std_hidden_norm |   mean_hidden_variance |   mean_cos_h_t_h_prev |   median_cos_h_t_h_prev |   percent_cos_gt_0_99 |   percent_cos_gt_0_95 |
|-------:|-------------------:|------------------:|-----------------------:|----------------------:|------------------------:|----------------------:|----------------------:|
|      1 |            3.90092 |          0.695034 |               0.241199 |              0.957096 |                0.976095 |              0.214087 |              0.751293 |
|      2 |            4.24915 |          0.84358  |               0.2922   |              0.968154 |                0.984887 |              0.325507 |              0.860326 |
|      3 |            3.72227 |          0.860523 |               0.226726 |              0.923904 |                0.95894  |              0.146439 |              0.543971 |
|      4 |            5.53311 |          0.714434 |               0.485122 |              0.972261 |                0.98583  |              0.421807 |              0.83844  |
|      5 |            4.68742 |          0.881705 |               0.354066 |              0.953884 |                0.975391 |              0.241544 |              0.680064 |

## Diagnostic 2: Prediction Equivalence
> This explains the behavior of the EXP025 architecture under the same training protocol.

| comparison_mode                |   top1_agreement_percent |   mean_abs_max_prob_difference |
|:-------------------------------|-------------------------:|-------------------------------:|
| shuffled_update                |                 0.992289 |                     0.0300549  |
| teacher_forced_eval_diagnostic |                 0.996343 |                     0.00868762 |
| zero_state_each_quote          |                 0.985771 |                     0.0598992  |

## Diagnostic 3: Scorer Hidden-State Weight Usage
|   seed |   candidate_block_weight_norm |   hidden_block_weight_norm |   similarity_weight_norm |   hidden_to_candidate_norm_ratio |   functional_mean_abs_logit_change_vs_zero |   functional_mean_abs_prob_change_vs_zero |
|-------:|------------------------------:|---------------------------:|-------------------------:|---------------------------------:|-------------------------------------------:|------------------------------------------:|
|      1 |                       2.95872 |                    2.4381  |                 0.462448 |                         0.824038 |                                  0.04261   |                                0.00152181 |
|      2 |                       3.02298 |                    2.52187 |                 0.416591 |                         0.834233 |                                  0.0454303 |                                0.00157679 |
|      3 |                       2.95684 |                    2.53079 |                 0.443338 |                         0.85591  |                                  0.0428066 |                                0.00186349 |
|      4 |                       2.90435 |                    2.72375 |                 0.397817 |                         0.937818 |                                  0.0468424 |                                0.00311493 |
|      5 |                       2.94297 |                    2.52534 |                 0.278953 |                         0.858093 |                                  0.0387128 |                                0.00160793 |

## Diagnostic 4: Previous-Speaker Probe
*Note: A hidden-only probe is structurally limited because all candidates receive the identical hidden state, yet only one is the true previous speaker. Meaningful comparisons examine whether hidden+candidate outperforms candidate-only.*
|   seed |   baseline_always_false_pr_auc |   probe_hidden_only_pr_auc |   probe_candidate_only_pr_auc |   probe_hidden_and_candidate_pr_auc |   probe_tf_tf_pr_auc |   probe_tf_ar_pr_auc |   baseline_always_false_bal_acc |   probe_hidden_and_candidate_bal_acc |   probe_candidate_only_bal_acc |
|-------:|-------------------------------:|---------------------------:|------------------------------:|------------------------------------:|---------------------:|---------------------:|--------------------------------:|-------------------------------------:|-------------------------------:|
|      1 |                        0.17606 |                   0.209477 |                      0.385304 |                            0.474828 |             0.507766 |             0.47465  |                             0.5 |                             0.697179 |                       0.653545 |
|      2 |                        0.17606 |                   0.202197 |                      0.409693 |                            0.48225  |             0.558966 |             0.445529 |                             0.5 |                             0.683511 |                       0.670605 |
|      3 |                        0.17606 |                   0.207973 |                      0.408431 |                            0.494013 |             0.607892 |             0.465542 |                             0.5 |                             0.697498 |                       0.67361  |
|      4 |                        0.17606 |                   0.202479 |                      0.41008  |                            0.469283 |             0.542349 |             0.469915 |                             0.5 |                             0.688012 |                       0.671857 |
|      5 |                        0.17606 |                   0.207173 |                      0.39378  |                            0.480855 |             0.575218 |             0.457007 |                             0.5 |                             0.683394 |                       0.657541 |

## Diagnostic 5: Gate Statistics
|   seed |   mean_update_gate |   std_update_gate |   mean_reset_gate |   std_reset_gate |   percent_update_gate_lt_0_1 |   percent_update_gate_gt_0_9 |
|-------:|-------------------:|------------------:|------------------:|-----------------:|-----------------------------:|-----------------------------:|
|      1 |           0.570105 |         0.0527254 |          0.490413 |       0.00564703 |                    0.0250087 |                    0.0733121 |
|      2 |           0.496058 |         0.0424876 |          0.479268 |       0.0161069  |                    0.0483655 |                    0.0557246 |
|      3 |           0.544605 |         0.0394778 |          0.531525 |       0.0161718  |                    0.0649655 |                    0.0560103 |
|      4 |           0.512828 |         0.0304839 |          0.580208 |       0.0184606  |                    0.0924645 |                    0.0304737 |
|      5 |           0.509668 |         0.0417329 |          0.546579 |       0.0310895  |                    0.0975383 |                    0.0611337 |

## Evidence Summary

| Dimension | Observation |
|---|---|
| **Hidden-state dynamics** | **Low Movement** (Cosine similarity remains high ~0.95; update gates hover at ~0.5 acting as a moving average) |
| **Prediction sensitivity** | **Negligible** (Zeroing the hidden state yields the exact same prediction >98.5% of the time) |
| **Probe increment over candidate-only** | **Substantial** (Adding the hidden state boosts PR-AUC by ~8 points from 0.40 to 0.48) |
| **Scorer integration** | **Limited evidence** (Weight norm exists, but functional probability change is <0.3% when state is zeroed) |

## Interpretation

EXP025C does not attempt to improve the GRU. It only provides a diagnostic baseline on why the EXP025 GRU failed to provide robust memory. 

**Conclusion:** The hidden state encodes useful speaker-history information, but the scorer fails to integrate it effectively for attribution. 

While the GRU successfully captured non-trivial discourse history (proven by the probe's substantial PR-AUC improvement), the downstream scoring MLP learned to completely ignore this recurrent state. The hidden state's functional impact on final probability outputs is practically zero, relying entirely on static candidate features.
