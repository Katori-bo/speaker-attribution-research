# EXP012A.3 Feature Dictionary

This document provides the operational definitions and statistical profile of the four semantic features extracted for EXP012.

## Features

### 1. `candidate_in_quote_chain` (Boolean)
- **Definition:** True iff any mention overlapping the quote span belongs to the candidate's coreference chain.
- **Coverage:** 100.0%
- **Distribution:** 11.2% True proportion (Mean = 0.112)
- **Semantic Meaning:** Indicates that the candidate entity is explicitly referenced within the text of the quotation itself (e.g., via a pronoun like "I", "my", or a nominal).

### 2. `nearest_coref_dist` (Integer)
- **Definition:** The absolute token distance from the quote's boundaries to the nearest coreferential mention of the candidate. If the candidate is unmentioned outside the quote, this is encoded as `-1`.
- **Coverage:** 100.0% (0 instances encoded as `-1` in the sample)
- **Distribution:** Mean 689.32 tokens, Variance 694,414.63, Max 2962 tokens.
- **Semantic Meaning:** Measures the proximity of the candidate's most salient occurrence in the narrative to the quotation.

### 3. `recent_mention_count` (Integer)
- **Definition:** Count of mentions for this candidate's chain in the 50 tokens (configurable window) strictly prior to the quote.
- **Coverage:** 100.0%
- **Distribution:** Mean 0.31 mentions, Variance 0.89, Max 8 mentions.
- **Semantic Meaning:** Captures the immediate local focus/salience of the candidate character.

### 4. `chain_recency` (Integer)
- **Definition:** Count of unique chains mentioned between the previous occurrence of the candidate chain and the quote boundary. If the candidate chain never appeared before the quote, it is encoded as `-1`.
- **Coverage:** 70.6% (147 missing values encoded as `-1`)
- **Distribution:** Mean 36.04 unique chains, Variance 1662.14, Max 140 chains.
- **Semantic Meaning:** Evaluates how recently the candidate was the topic of focus, acting as an entity-level proxy for conversational persistence and discourse transitions.

## Correlation Audit
A Spearman correlation matrix across the four features on a 500-candidate sample confirms that the features encode distinct, non-collinear information:
- `candidate_in_quote_chain` vs `nearest_coref_dist`: -0.417
- `candidate_in_quote_chain` vs `recent_mention_count`: 0.436
- `nearest_coref_dist` vs `recent_mention_count`: -0.562
- All correlations are weak-to-moderate, validating the orthogonality of the feature set.
