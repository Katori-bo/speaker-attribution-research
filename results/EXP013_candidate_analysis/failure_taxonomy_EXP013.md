# EXP013 Failure Taxonomy & Candidate Analysis

Based on an analysis of the `UNCHANGED_FAILURE` and `REGRESSION` quotes from EXP012B, the remaining Type A errors (where entity mapping succeeded but the prediction failed) fall into the following four primary categories:

### 1. Speaker-Addressee Flow & Turn-Taking
**Example:** `PrideAndPrejudice_149`
> **Speaker 1:** "They are wanted in the farm, Mr. Bennet, are not they?"
> **Speaker 2 (Mr. Bennet):** "They are wanted in the farm much oftener than I can get them."

**Analysis:** Speaker 1 explicitly addresses Speaker 2 ("Mr. Bennet"). In the subsequent quote, Speaker 2 responds. Our current state tracker (`ConversationStateModule`) tracks recent speakers, but has no concept of *who was addressed*.
**Missing Capability:** Speaker-Addressee reasoning. If a candidate is explicitly addressed in $Quote_{t-1}$, they are highly likely to be the speaker of $Quote_t$.

---

### 2. Attribution Tag Competition
**Example:** `AlicesAdventuresInWonderland_72` (Regression)
> **Quote:** "You are not attending!" said the Mouse to Alice severely.

**Analysis:** Both the actual speaker ("Mouse") and the addressee ("Alice") are explicitly present in the attribution tag adjacent to the quote. Our semantic feature `nearest_coref_dist` measures distance, but when multiple characters are equally close to the quote boundaries, the model gets confused and picks the one with higher global salience (Alice).
**Missing Capability:** Structural parsing. Distance alone cannot disambiguate `"said A to B"`.

---

### 3. Nominal Coreference Linking
**Example:** `PrideAndPrejudice_831` and `DaisyMiller_45`
> **Context:** "...was grateful to her uncle for saying,"
> **Quote:** "There are very few people of whom so much can be said."

> **Context:** "...cried the child."
> **Quote:** "Her name is Daisy Miller!"

**Analysis:** The speaker is identified in the text by a nominal ("her uncle", "the child"). If BookNLP's coreference engine failed to link these nominals to their canonical character chains (Mr. Gardiner, Randolph), our pipeline has no way to rank them highly.
**Missing Capability:** Stronger entity resolution for nominals, or leveraging direct BookNLP attribution outputs rather than relying strictly on the coreference cluster.

---

### 4. First-Person Monologue Saturation
**Example:** `PrideAndPrejudice_458` (Mr. Collins' Proposal)
> **Quote:** "My reasons for marrying are, first, that I think it a right thing... (massive paragraph mentioning Lady Catherine, Miss de Bourgh, Mr. Bennet, etc.)"

**Analysis:** In massive monologues, the speaker refers to themselves as "I" or "my", while frequently naming other characters. The `candidate_in_quote_chain` feature will highly rank whoever is explicitly mentioned inside the quote. Unless "I" is perfectly resolved to the speaker's coreference chain, the speaker gets drowned out by the other entities mentioned in the text.
**Missing Capability:** High-confidence 1st-person pronoun resolution inside quote boundaries.

---

## Next Steps: Choosing EXP013

Based on this taxonomy, we need to select the next capability to implement and evaluate.

The two strongest candidates for EXP013 are:

**Candidate A: Speaker-Addressee Reasoning**
Extracting explicit vocatives (e.g. "Mr. Bennet, are not they?") to boost the target's probability in the subsequent turn.

**Candidate B: Explicit Attribution Extraction**
Rather than relying on `nearest_coref_dist` to guess who is in the attribution tag, we could parse the tag (e.g., using BookNLP's native attribution features, or a dependency parse) to differentiate the subject ("said A") from the object ("to B").
