# EXP014B.0b Attribution Oracle Validation

## Objective
Verify that the entity extracted by the naive regex attribution parser actually matches the true gold speaker for the quote, ensuring that the 40% feasibility number represents true recovery potential and not just noisy availability.

## Results
- **Resolvable attribution tags:** 116
- **Correct speaker matches:** 46
- **Precision:** 39.7%

## Error Analysis Examples
The naive regex approach (looking within a +/- 150 char window) frequently associates the quote with the wrong entity. Common failure modes:
1. **Cross-quote interference:** Finding an attribution tag that belongs to a neighboring quote (e.g., matching "said Darcy" when the current quote is spoken by Elizabeth in the next sentence).
2. **Attribution Object confusion:** Matching the recipient instead of the speaker due to proximity.
3. **Pronoun ambiguity:** Assuming pronouns resolve correctly when they often don't without full coreference.
4. **Name fragment extraction:** Extracting "Mrs" instead of "Mrs. Bennet" or "the Rabbit" instead of "Pat".

## Conclusion
Extraction is highly noisy. A precision of ~40% is below the acceptable threshold. A purely naive proximity regex is insufficient to accurately identify the grammatical speaker of a specific quote. A more structurally aware parser is needed to avoid injecting harmful noise into the model.
