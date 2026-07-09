# ADR 010: Reject Static Lexical Fingerprints

## Date
2026-07-09

## Status
Accepted

## Context
In EXP019A, we tested the hypothesis that speakers could be identified by unique lexical fingerprints (TF-IDF vectors of their historical utterances). We compared a baseline autoregressive model (EXP014 AR) against models that incorporated these lexical fingerprints.

## Decision
We reject the use of static TF-IDF lexical fingerprints for speaker attribution.

## Rationale
The statistical validation of EXP019A results shows that static lexical fingerprints do not provide measurable complementary information:
- **No Significant Improvement**: The overall accuracy delta was negative (-0.0028) with a 95% confidence interval crossing zero ([-0.009, +0.004]). The McNemar test p-value (0.43) confirms the difference is not statistically significant.
- **Gold Fingerprint Ceiling Negligible**: Even when using gold fingerprints (the theoretical ceiling), performance improvement was negligible.
- **Identity Shuffle Unchanged**: Shuffling the identities of the fingerprints resulted in identical performance (65.18%), proving the model did not learn to map specific fingerprint features to specific characters.
- **Feature Ablation**: Feature ablation showed that removing these lexical features removed no useful signal from the model.

## Allowed Conclusion
Static TF-IDF lexical fingerprints do not provide measurable complementary information to the current HistGBM model.

## Forbidden Conclusion
This does not mean that characters do not have unique styles, only that this specific static TF-IDF fingerprint method failed to capture or utilize that style in a way that improved upon existing features.
