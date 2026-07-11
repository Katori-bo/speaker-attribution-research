# EXP026B Preregistered Plan

## Research Question

Can explicit auxiliary supervision make the GRU hidden state encode speaker-history information in a way that improves autoregressive attribution?

## Hypotheses

The GRU currently learns weak or poorly organized history representations because attribution loss alone does not force the hidden state to preserve the previous-speaker relation. Adding an auxiliary previous-speaker objective regularizes the hidden state toward a discourse variable known to be important.

## Variants

- **Variant A**: Candidate-only no-memory baseline (`EXP026ACandidateOnlyScorer`)
- **Variant B**: Bilinear GRU baseline, no auxiliary head / loss (`EXP026ABilinearSpeakerGRU` from EXP026A)
- **Variant B2**: Bilinear GRU with auxiliary head present, loss deactivated ($\lambda = 0$)
- **Variant C**: Bilinear GRU + auxiliary loss with validation-selected $\lambda \in \{0.1, 0.3, 1.0\}$

## Auxiliary Objective

At quote step $t$, use the hidden state $h_t$ before updating to predict whether each candidate is the previous gold speaker of the sequence timeline:
$$
\text{aux\_score}(c, h) = c^T W_{aux} h
$$
Target is 1 if candidate $c_i == \text{gold\_speaker\_id}[t-1]$, else 0.
We apply masked candidate-level cross-entropy loss over quotes where the previous gold speaker is available and is present in the current candidate set.

$$
L = L_{attribution} + \lambda L_{aux}
$$

## Acceptance Criteria

1. **Predictive Success**: Selected $C_{\lambda} > A$ by $\ge +0.5$ pp mean accuracy, winning on $\ge 3/5$ seeds.
2. **Incremental Success**: Selected $C_{\lambda} > B$ by $\ge +0.5$ pp mean accuracy, winning on $\ge 3/5$ seeds.
3. **No Slice Regressions**: Neither implicit nor anaphoric mean accuracy may decrease by more than **$0.5$ pp** relative to A or B.
4. **Causal Memory**:
   - $\text{Normal} - \text{Zero-State} \ge 0.5$ pp.
   - $\text{Normal} - \text{Shuffled-Update} \ge 0.25$ pp.
   - $\text{Teacher-Forced}$ must not be worse than $\text{Normal}$ by $> 0.5$ pp.
5. **Recovery/Regression**: For both $C-A$ and $C-B$, recoveries must exceed regressions overall.

## Retirement Pre-registration

If EXP026B fails both predictive and causal criteria, retire the feature-only GRU line and move to quote semantic representations (EXP027).

## Commands

```bash
.venv/bin/python -m pytest tests/test_exp026b_auxiliary_supervision.py
CPU_TEST_RUN=1 .venv/bin/python scripts/run_EXP026B_auxiliary_supervision.py
.venv/bin/python scripts/run_EXP026B_auxiliary_supervision.py
```
