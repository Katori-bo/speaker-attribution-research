# EXP003: Symbolic Attribution Rules Report

- **Total Quotes:** 37131
- **Candidate Generator Ceiling:** 92.76%
- **Rule Oracle Accuracy:** 76.26%
- **Engine Final Accuracy:** 66.53%
- **Abstentions (No Rule Fired):** 1 (0.00%)

## Quote-Type Error Breakdown
| Quote Type | Total | Correct | Accuracy |
|------------|-------|---------|----------|
|  | 14 | 6 | 42.86% |
| Anaphoric | 9202 | 4571 | 49.67% |
| Explicit | 11185 | 7114 | 63.60% |
| Implicit | 16716 | 13007 | 77.81% |
| nan | 14 | 5 | 35.71% |

## Rule Quality & Contribution
| Rule | Applicable | Fired | Precision (Fired) | Won | Contribution (Won) |
|------|------------|-------|-------------------|-----|--------------------|
| Explicit Attribution | 33801 | 3333 | 94.24% (3141/3333) | 3333 | 94.24% (3141/3333) |
| Dialogue Alternation | 37075 | 31605 | 68.43% (21626/31605) | 28793 | 69.20% (19924/28793) |
| Previous Speaker | 37103 | 37103 | 14.75% (5474/37103) | 4978 | 32.90% (1638/4978) |
| Nearest Mention | 36783 | 36783 | 0.58% (215/36783) | 26 | 0.00% (0/26) |

## Major Rule Conflicts
| Rule A | Rule B | Count |
|--------|--------|-------|
| Dialogue Alternation | Previous Speaker | 31605 |
| Dialogue Alternation | Nearest Mention | 27372 |
| Nearest Mention | Previous Speaker | 14632 |
| Explicit Attribution | Nearest Mention | 3170 |
| Explicit Attribution | Previous Speaker | 2962 |
| Dialogue Alternation | Explicit Attribution | 1145 |
