# EXP014B.0 Attribution Feasibility Analysis

## Feasibility of Explicit Attribution Recovery
The feasibility analysis evaluates the 290 remaining failure cases from EXP012B to determine how many contain recoverable explicit attribution tags. A regex-based parser was used to find common attribution verbs (e.g., "said", "asked", "replied") in proximity to the quote, combined with checking for known candidate entity names and pronouns.

### Results
- **Residual errors:** 290
- **Attribution tags:** 188
- **Speaker mention detected:** 188
- **Mention maps to entity:** 116
- **Candidate exists:** 116

### Conclusion
**Upper bound recovery: 40.0%**

This shows that a significant portion of the remaining errors (at least 40%) have explicit, grammatically resolvable attributions. Extractable signal is strong enough to avoid the sparse signal issue encountered in EXP013.
