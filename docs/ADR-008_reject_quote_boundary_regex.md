# ADR-008: Reject Quote-Boundary Regex Attribution Extractor

## Context
During EXP014B.0c (Structured Attribution Feasibility), we attempted to resolve speaker roles for quotes by anchoring regular expression matching to the boundaries of the quote (the immediately preceding or following 80 characters). The goal was to avoid noise from neighboring quotes.

## Decision
We reject the quote-boundary regex attribution extractor as a source for explicit attribution representations.

## Reason
The extractor yielded a precision of only 64.3%. While it improved upon the naive proximity regex approach (which had 39.7% precision), it is still too low to function as a reliable explicit signal.

The primary failure modes observed were:
- **Subject/Object Confusion:** The string pattern cannot differentiate between the syntactic subject ("said Darcy") and the object/recipient ("said to Darcy" or "said Darcy to Elizabeth").
- **Pronoun Attachment Ambiguity:** Guessing the reference of a pronoun purely based on string heuristics and gender mappings leads to unacceptably high error rates without full coreference.
- **Nominal Speaker Phrases:** The regex failed to properly bound nominal subjects like "the young man" versus extraneous descriptive text.

## Consequences
- Do not create attribution features using regex extraction.
- Proceed to test syntactic representation sources (such as a dependency parser) to determine if structural, grammar-aware extraction resolves the subject/object and boundary confusions.
