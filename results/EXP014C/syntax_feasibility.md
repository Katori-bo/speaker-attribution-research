# EXP014C.0 Syntax Attribution Feasibility

## Objective
Evaluate whether dependency parsing (extracting `nsubj` or `nsubjpass` of speech verbs) can reliably recover speaker roles from explicit attribution structures, avoiding the subject/object confusion seen in string matching.

## Results
- **Total residual errors:** 290
- **Syntax attribution found:** 201

### Track A: Named Speakers
- **Count:** 121
- **Precision:** 23.1%

### Track B: Pronoun Speakers
- **Count:** 80
- **Precision:** 70.0%

- **Overall oracle recovery:** 29.0%

## Analysis
The syntax parser successfully identified grammatical subjects (`nsubj`), solving the subject/object confusion (e.g., distinguishing "Darcy" from "Elizabeth" in "Darcy said to Elizabeth"). However, because the parser was run on a +/- 150 character window and selected the "closest" speech verb, it suffered heavily from the **cross-quote contamination** problem. 

The closest speech verb often belongs to a neighboring quote, leading the parser to extract the perfect syntactic subject of the *wrong* attribution tag (e.g., extracting "the Hatter" as the speaker for a quote that belongs to "the King", because "the Hatter said" was nearby).

## Decision Gate
The named attribution precision (23.1%) is far below the required 85% threshold. 
Therefore, this extraction method cannot be used as-is. We must reject proceeding to EXP014C.1 without a tighter coupling between the quote and the speech verb.
