# ADR 009: Accept Quote-Aligned Named Syntactic Attribution

## Status
Accepted

## Representation
Quote-aligned named syntactic attribution

## Context
In EXP014, we explored whether explicit modeling of grammatical speaker roles could recover failures where entity salience or coreference assigns the wrong dialogue role. We iteratively tested several extraction algorithms:
1. Naive proximity regex (±150 chars) - Rejected (39.7% precision, cross-quote contamination).
2. Quote-boundary regex - Rejected (64.3% precision, subject/object confusion).
3. Free-window dependency syntax - Rejected (role confusion solved, but cross-quote contamination remained).
4. Quote-aligned dependency syntax (constrained to immediate boundaries) - Accepted.

## Reason
Although the coverage of quote-aligned syntactic attribution is limited (~12.9% globally), the representation introduces a highly reliable grammatical speaker signal (99.7% precision).

It improves:
- Accuracy (+0.20%)
- PR-AUC (0.9155 → 0.9162)
- LogLoss (0.2126 → 0.2096)
- Confidence calibration (Gold probability moved from 0.979 to 0.993 when active)

While introducing:
- **0 regressions**

EXP014 demonstrates that high-precision explicit attribution signals provide complementary information primarily through confidence calibration rather than massive top-1 recovery.

## Scope
- Only PROPN syntactic subjects (nsubj) attached to valid speech verbs are accepted.
- Pronouns and nominal attribution remain strictly outside scope to avoid coreference conflation.

## Important Scientific Observation
EXP014 creates an interesting contrast with EXP013 (Sparse Addressee State).
- **EXP013** used a broad signal (interaction history, moderate coverage) but had weak construct validity, leading to its rejection.
- **EXP014** used a narrow signal (explicit grammatical speaker, low coverage) but had very high construct validity and precision, leading to its acceptance.
This supports the broader research claim: in lightweight speaker attribution, representation precision may matter more than representation breadth.
