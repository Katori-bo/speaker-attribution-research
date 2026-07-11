# EXP024 Position Feature Experiment

## Core Metrics
| Condition | Overall Acc | Implicit Acc | Anaphoric Acc | MRR | LogLoss |
|-----------|-------------|--------------|---------------|-----|---------|
| nomemory_baseline | 71.46% | 65.01% | 60.42% | 0.8113 | 0.7790 |
| nomemory_plus_position_index | 72.69% | 63.43% | 62.34% | 0.8268 | 0.7540 |
| nomemory_plus_position_bucket | 72.81% | 63.96% | 62.34% | 0.8252 | 0.7568 |
| nomemory_plus_shuffled_position_index | 71.10% | 63.56% | 61.54% | 0.8125 | 0.7735 |
| nomemory_plus_shuffled_position_bucket | 70.63% | 62.91% | 60.10% | 0.8102 | 0.7764 |
| gru_baseline | 73.29% | 70.25% | 61.38% | 0.8225 | 0.7510 |
| gru_plus_position_index | 71.46% | 61.21% | 60.10% | 0.8161 | 0.7926 |
| gru_plus_position_bucket | 71.94% | 61.99% | 60.42% | 0.8177 | 0.7799 |

## McNemar Statistical Tests
- **nomemory_plus_position_index vs nomemory_baseline (nomemory_plus_position_index vs nomemory_baseline)**: 5.0349e-02
- **nomemory_plus_position_bucket vs nomemory_baseline (nomemory_plus_position_bucket vs nomemory_baseline)**: 3.4634e-02
- **gru_plus_position_index vs gru_baseline (gru_plus_position_index vs gru_baseline)**: 1.7706e-02
- **gru_plus_position_bucket vs gru_baseline (gru_plus_position_bucket vs gru_baseline)**: 7.4353e-02
- **nomemory_plus_shuffled_position_index vs nomemory_baseline (nomemory_plus_shuffled_position_index vs nomemory_baseline)**: 5.2317e-01

## Decision Rule Interpretation
> **Conclusion**: Position helps No-Memory but not GRU. GRU already captures some salience structure natively.