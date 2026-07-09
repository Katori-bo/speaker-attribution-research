# EXP014B.0c Structured Attribution Oracle Validation

## Objective
Evaluate whether quote-boundary-aware attribution extraction (anchoring search to the start of the right context or end of the left context within an 80 character window) yields a high-precision speaker signal compared to the naive +/- 150 char proximity approach.

## Results
- **Total EXP012B failures:** 290
- **Extracted tags:** 94
- **Resolvable attribution tags:** 56
- **Correct speaker matches:** 36
- **Precision:** 64.3%

## Analysis
Applying structural constraints (only looking immediately preceding or following the quote boundary) significantly reduced the false positive rate (Coverage dropped from 116 resolvable tags down to 56). 

However, the precision only increased from 39.7% to 64.3%. 

### Remaining Error Modes
Even with strict boundary constraints, several errors persist:
1. **Pronoun ambiguity:** The heuristic for matching pronouns to gold entities (guessing gender) is fundamentally flawed without true coreference. Instances where "she" is extracted for a quote spoken by a male entity (or an unmapped entity) still occur.
2. **Object vs Subject Confusion within boundaries:** E.g. extracted `to Elizabeth` or `in an undertone to the Queen`. The regex `verbs \s+ name` doesn't differentiate between "said Darcy" and "said to Darcy".
3. **Conversational turn complexity:** In rapid dialogue, an attribution tag might immediately follow a quote but actually describe an action of the listener rather than the speaker.

## Conclusion
The structured boundary approach improved precision, but **64.3% still falls below the required 70% threshold** set for the decision gate. 

Therefore, simple string matching (even with strict positional anchors) is insufficient to extract this representation reliably. We must consider a different representation source or parsing strategy (like dependency parsing).
