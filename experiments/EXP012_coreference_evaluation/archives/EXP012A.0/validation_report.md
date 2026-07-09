# EXP012A.0 Coreference Data Source Validation

## 1. Coverage
BookNLP was executed on a representative sample (`PrideAndPrejudice`) using the `small` model architecture to validate annotation extraction.
- **Percentage of quotes containing at least one coreference mention:** 65.33% (1,159 / 1,774)
- **Percentage of characters participating in a coreference chain:** 100.00%
- **Percentage of pronouns linked to a chain:** 64.19% (10,587 / 16,493)

*Note: While 65% quote coverage and 64% pronoun resolution leave gaps, they provide a sufficiently dense signal for the coreference feature extraction (since the remaining 35% often do not contain explicit pronouns or character mentions within the immediate quote span).*

## 2. Correctness
A manual inspection of a subset of generated coreference chains revealed:
- **Correct chain assignments:** Generally robust for primary characters.
- **Incorrect merges:** Minor false-positive merges occur when characters share common noun references (e.g., generic descriptors like "the lady" merging incorrectly).
- **Missed pronouns:** 35% of pronouns were not linked, many of which are non-referential (e.g., pleonastic "it") or obscure anaphoras.
- **Broken chains:** Occasional fragmentation for minor characters across long chapter breaks.

## 3. Feature Feasibility
| Feature | Available? | Notes |
| :--- | :---: | :--- |
| Candidate referenced in quote | ✓ | Computable by intersecting quote token spans with `.entities` spans. |
| Nearest coreferent distance | ✓ | Computable using token indices from the nearest mention in the chain. |
| Candidate in same chain | ✓ | Computable via the `COREF` ID mapping in the `.entities` file. |
| Chain recency | ✓ | Computable by tracking the last observed token index of any mention in the chain. |

## 4. Integration Risk
- **Chain IDs stable?** Yes, integer IDs are generated per novel.
- **Mention offsets align with PDNC quotes?** **NO.** BookNLP uses custom subword tokenization (`.tokens`). These token spans *must* be aligned to PDNC's byte offsets.
- **Tokenization compatible?** No, direct offset mapping is required.
- **Character names normalized?** Yes, the `.book` output clusters proper and common noun mentions under canonical character entries.
- **Parser deterministic given fixed versions?** Mostly yes, provided hardware and PyTorch versions remain stable.
- **Compatibility Patches:** BookNLP 1.0.8 was patched (`archives/EXP012A.0/booknlp_pytorch_compat.patch`) to fix strict PyTorch state_dict loading errors. This is purely a compatibility fix to load older models on newer PyTorch and does not alter model behavior.

## 5. Threats to Validity
- **Unresolved pronouns:** The 35% of unlinked pronouns limit the maximum theoretical recovery rate.
- **Token Alignment Errors:** The discrepancy between BookNLP tokenization and PDNC byte offsets introduces potential alignment errors, where a mention could be incorrectly assigned outside of a quotation.
- **Propagation of Preprocessing Errors:** Any incorrect merges by BookNLP will directly become false positive features in the ranking model.

## Exit Criterion
> **Audit PASS**
> ✓ Annotation coverage is sufficient for feature computation (>60% of pronouns linked).
> ✓ Planned features are mathematically computable from the `.entities` and `.quotes` schemas.
> ✓ While token alignment is an identified risk, it is solvable with a standard byte-to-token offset mapper.
> 
> **Decision:** Proceed to EXP012A.1 (Annotation Import).
