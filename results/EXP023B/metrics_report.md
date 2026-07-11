# EXP023B Entity Binding Interaction Analysis

## Core Metrics
| Condition | Overall Acc | Implicit Acc | Anaphoric Acc | MRR | LogLoss |
|-----------|-------------|--------------|---------------|-----|---------|
| nomemory_no_anchor | 71.46% | 65.01% | 60.42% | 0.8113 | 0.7790 |
| nomemory_persistent_anchor | 71.14% | 68.41% | 56.73% | 0.8162 | 0.7867 |
| gru_no_anchor | 73.29% | 70.25% | 61.38% | 0.8225 | 0.7510 |
| gru_persistent_anchor | 66.49% | 56.49% | 52.72% | 0.7849 | 0.8533 |

## McNemar Statistical Tests
- **GRU Persistent Anchor vs GRU No Anchor (gru_persistent_anchor vs gru_no_anchor)**: 4.5548e-18
- **GRU Persistent Anchor vs Nomemory Persistent Anchor (gru_persistent_anchor vs nomemory_persistent_anchor)**: 9.5134e-11
- **GRU Persistent Anchor vs Nomemory No Anchor (gru_persistent_anchor vs nomemory_no_anchor)**: 3.1890e-11
- **Nomemory Persistent Anchor vs Nomemory No Anchor (nomemory_persistent_anchor vs nomemory_no_anchor)**: 6.5406e-01
- **GRU No Anchor vs Nomemory No Anchor (gru_no_anchor vs nomemory_no_anchor)**: 8.3634e-06

## Decision Rule Interpretation
> **Conclusion**: The GRU-anchor architecture is not justified.