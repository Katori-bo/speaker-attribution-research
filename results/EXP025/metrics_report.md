# EXP025 GRU Stability and Memory Ablation

## EXP025A: Seed Stability

| Model              |   Accuracy_Mean |   Accuracy_Std |   Accuracy_Min |   Accuracy_Max |   Accuracy_95CI |   Implicit_Mean |   Anaphoric_Mean |
|:-------------------|----------------:|---------------:|---------------:|---------------:|----------------:|----------------:|-----------------:|
| nomemory_no_anchor |        0.719714 |      0.0082581 |       0.709459 |       0.732114 |      0.00723855 |        0.659764 |         0.609936 |
| gru_normal         |        0.699762 |      0.0188207 |       0.683227 |       0.728935 |      0.0164971  |        0.604718 |         0.593269 |

GRU wins on 0/5 seeds.
> **Conclusion**: The GRU improvement is not stable enough to treat as a reliable architectural gain.

## EXP025B: GRU Memory Ablation

- `gru_normal`: 0.6998
- `gru_reset_hidden_each_quote`: 0.6995
- `gru_zero_update`: 0.6995
- `gru_shuffled_update`: 0.7007
- `gru_teacher_forced_eval_diagnostic`: 0.7003

> **Conclusion**: The gain comes from the neural candidate encoder/scorer, not recurrent memory.

*Note: `gru_zero_update` and `gru_reset_hidden_each_quote` produced nearly identical results because scoring occurs before the memory update, meaning both effectively evaluate each quote from a zero state.*