# EXP023 Entity Binding Analysis

## Core Metrics
| Condition | Overall Acc | Implicit Acc | Anaphoric Acc | MRR | LogLoss |
|-----------|-------------|--------------|---------------|-----|---------|
| no_anchor | 71.46% | 65.01% | 60.42% | 0.8113 | 0.7790 |
| constant | 71.03% | 62.25% | 61.86% | 0.8089 | 0.7806 |
| position | 74.13% | 68.55% | 62.34% | 0.8335 | 0.7548 |
| ephemeral | 69.24% | 60.03% | 56.25% | 0.8034 | 0.7856 |
| unstable | 70.47% | 62.78% | 58.65% | 0.8078 | 0.7928 |
| frozen_persistent | 71.14% | 68.41% | 56.73% | 0.8162 | 0.7867 |
| deterministic_hash | 68.40% | 60.55% | 54.49% | 0.7959 | 0.8530 |
| trainable_persistent | 72.06% | 67.23% | 60.42% | 0.8197 | 0.7908 |
| shuffled_persistent | 69.08% | 62.12% | 53.53% | 0.8019 | 0.8140 |

## Anchor Diagnostics
| Condition | Persistence Score | Unique Anchors | Trainable Params |
|-----------|-------------------|----------------|------------------|
| no_anchor | 0.0000 | 1 | char_emb.weight: frozen, pos_emb.weight: grad, scorer.0.weight: grad, scorer.0.bias: grad, scorer.2.weight: grad, scorer.2.bias: grad, scorer.4.weight: grad, scorer.4.bias: grad |
| constant | 1.0000 | 1 | constant_vector: frozen, char_emb.weight: frozen, pos_emb.weight: grad, scorer.0.weight: grad, scorer.0.bias: grad, scorer.2.weight: grad, scorer.2.bias: grad, scorer.4.weight: grad, scorer.4.bias: grad |
| position | 0.6131 | 15 | char_emb.weight: frozen, pos_emb.weight: grad, scorer.0.weight: grad, scorer.0.bias: grad, scorer.2.weight: grad, scorer.2.bias: grad, scorer.4.weight: grad, scorer.4.bias: grad |
| ephemeral | -0.0014 | 13166 | char_emb.weight: frozen, pos_emb.weight: grad, scorer.0.weight: grad, scorer.0.bias: grad, scorer.2.weight: grad, scorer.2.bias: grad, scorer.4.weight: grad, scorer.4.bias: grad |
| unstable | 0.0014 | 1208 | char_emb.weight: frozen, pos_emb.weight: grad, scorer.0.weight: grad, scorer.0.bias: grad, scorer.2.weight: grad, scorer.2.bias: grad, scorer.4.weight: grad, scorer.4.bias: grad |
| frozen_persistent | 1.0000 | 134 | char_emb.weight: frozen, pos_emb.weight: grad, scorer.0.weight: grad, scorer.0.bias: grad, scorer.2.weight: grad, scorer.2.bias: grad, scorer.4.weight: grad, scorer.4.bias: grad |
| deterministic_hash | 1.0000 | 134 | char_emb.weight: frozen, pos_emb.weight: grad, scorer.0.weight: grad, scorer.0.bias: grad, scorer.2.weight: grad, scorer.2.bias: grad, scorer.4.weight: grad, scorer.4.bias: grad |
| trainable_persistent | 1.0000 | 134 | char_emb.weight: grad, pos_emb.weight: grad, scorer.0.weight: grad, scorer.0.bias: grad, scorer.2.weight: grad, scorer.2.bias: grad, scorer.4.weight: grad, scorer.4.bias: grad |
| shuffled_persistent | 1.0000 | 134 | char_emb.weight: frozen, pos_emb.weight: grad, scorer.0.weight: grad, scorer.0.bias: grad, scorer.2.weight: grad, scorer.2.bias: grad, scorer.4.weight: grad, scorer.4.bias: grad |

## McNemar Statistical Tests
### Vs Frozen Persistent Random Anchor
- **no_anchor**: 6.5406e-01
- **constant**: 9.0072e-01
- **position**: 1.0126e-05
- **ephemeral**: 5.1291e-03
- **unstable**: 3.3812e-01
- **deterministic_hash**: 8.8749e-05
- **trainable_persistent**: 1.3351e-01
- **shuffled_persistent**: 2.3894e-03

### Vs No Anchor Baseline
- **constant**: 2.3531e-01
- **position**: 5.0274e-05
- **ephemeral**: 3.8489e-04
- **unstable**: 1.3890e-01
- **frozen_persistent**: 6.5406e-01
- **deterministic_hash**: 1.4408e-05
- **trainable_persistent**: 3.8435e-01
- **shuffled_persistent**: 3.8321e-04

### Vs MLP CE Baseline (EXP021A.2)
- **no_anchor**: 9.7852e-05
- **constant**: 5.3087e-04
- **position**: 8.6699e-13
- **ephemeral**: 5.7068e-01
- **unstable**: 1.8697e-02
- **frozen_persistent**: 1.6844e-03
- **deterministic_hash**: 6.3526e-01
- **trainable_persistent**: 1.9629e-05
- **shuffled_persistent**: 7.3532e-01

## Success Criteria
**Frozen Persistent - Ephemeral >= 1.5 pp**: PASS (1.91 pp)