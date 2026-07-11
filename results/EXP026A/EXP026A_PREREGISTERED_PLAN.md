# EXP026A Preregistered Plan

## Question

Does an explicit bilinear candidate-history compatibility function convert the GRU's decodable historical signal into measurable, causally memory-dependent attribution improvement?

## Frozen Scorer

Variant C uses exactly:

```text
s(c,h) = f(c) + c^T W h
```

where `f(c)` is the full candidate-only scoring branch used by the EXP026A no-memory control, and the interaction branch has no bias. Only `W` is trainable in the interaction term. Therefore `s(c,0)=f(c)` by construction.

No auxiliary objective, candidate-feature dropout, gate, residual MLP, hidden-state bias, attention, deeper fusion, or alternate scorer is part of EXP026A.

## Frozen Stopping Rule

Regardless of statistical significance, EXP026A is the final scorer-integration experiment for the current feature-only GRU architecture.

## Acceptance Rules

- Variant C must improve over the five-seed no-memory baseline of 71.97% by at least +0.5 pp.
- Paired seed delta must be positive, and Variant C must win on most seeds.
- No major implicit or anaphoric subset regression may occur.
- Normal autoregressive evaluation must exceed zero-state by at least +0.5 pp.
- Normal autoregressive evaluation must exceed shuffled-update evaluation.
- Teacher-forced evaluation should not be materially or consistently worse than normal autoregressive evaluation.
- Probe success alone cannot override failed attribution results.
- If `c^T W h` is near zero almost everywhere, do not interpret any accuracy gain as memory interaction.

## Commands

```bash
pytest tests/test_exp026a_bilinear_interaction.py
CPU_TEST_RUN=1 python scripts/run_EXP026A_bilinear_memory_interaction.py
python scripts/run_EXP026A_bilinear_memory_interaction.py
```
