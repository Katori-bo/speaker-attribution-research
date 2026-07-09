import pandas as pd
import os

df = pd.read_csv("results/EXP010/semantic_annotations_master.csv")

missing_df = df[df['ANNOTATION: Primary Category'].isna()]

missing_count = len(missing_df)
print(f"Missing quotes count: {missing_count}")

output_file = "annotations/missing_quotes_blind.md"

manual_summary = """
# EXP010 Semantic Error Taxonomy (Coding Manual Summary)
1a. Reference: Pronominal Coreference - pronoun or discourse marker requires reasoning to resolve to antecedent.
1b. Reference: Lexical Normalization / Alias Matching - named alias, title, or spelling variation fails exact matching (solvable by alias dictionary).
2. Discourse: Speaker Continuity - no referring expression, implicitly understood by structure (A-B-A-B, topic continuation).
3. Pragmatics: Speaker–Addressee Semantics - internal semantic clues about who is spoken TO implies who is speaking.
4. Discourse Structure: Scene Transitions - speaker changes because physical scene/context changes.
5. Other / Unclassified - novel phenomenon.

Evidence (Controlled Vocabulary): pronoun, alias, discourse marker, paragraph break, scene break, explicit addressee, dialogue tag absent, topic continuation, narrative focus, other.
Context Window Needed: Local (same sentence), Nearby (same paragraph), Conversation, Scene, Long-range, Unknown.
Explicit Alternative Feasible?: Yes (solvable without semantic modeling), No, Unsure.
"""

prompt = """
---
# Annotation Prompt

You are acting as an independent linguistic annotator.

Follow the EXP010 Semantic Error Taxonomy exactly.

Your task is NOT to identify every linguistic phenomenon.

Instead identify the PRIMARY reason why the evaluated explicit contextual representation would fail.

Do not use model predictions.

For every quote output exactly:

**Quote ID:** [Insert ID here]
Primary Category:
Secondary Category:
Evidence:
Context Window Needed:
Confidence:
Explicit Alternative Feasible?:
Notes:

Keep Notes to 1-2 sentences explaining WHY the explicit representation failed.

If multiple phenomena exist, choose only one Primary Category.

Do not skip quotes.
"""

version_a_cols = [
    "Quote_ID", "Failure_Category", "Gold_Speaker", "Candidate_List",
    "Quote", "Context (Referring Expression)", "Explicit Signals Present?",
    "Previous Speaker", "Conversation Turn Index", "Discourse Context Length"
]

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"# EXP010 Missing Quotes (BLIND)\n\n")
    f.write(f"Total Missing Quotes: {missing_count}\n\n")
    f.write(manual_summary + "\n")
    
    for idx, row in missing_df.iterrows():
        f.write(f"## Quote {idx + 1}\n")
        for col in version_a_cols:
            f.write(f"**{col}:** {row[col]}\n")
        f.write("\n")
        
    f.write(prompt)

print(f"Generated {output_file} with the missing quotes ready for copy-pasting.")
