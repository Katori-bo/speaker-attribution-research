import pandas as pd

df = pd.read_csv("results/EXP010/semantic_annotations_master.csv")
missing_df = df[df['ANNOTATION: Primary Category'].isna()]

missing_count = len(missing_df)

output_file = "annotations/missing_40_quotes_blind.md"

manual_summary = """
# EXP010 Semantic Error Taxonomy (Coding Manual Summary)
1a. Reference: Pronominal Coreference
1b. Reference: Lexical Normalization / Alias Matching
2. Discourse: Speaker Continuity
3. Pragmatics: Speaker–Addressee Semantics
4. Discourse Structure: Scene Transitions
5. Other / Unclassified

Evidence: pronoun, alias, discourse marker, paragraph break, scene break, explicit addressee, dialogue tag absent, topic continuation, narrative focus, other.
Context Window Needed: Local, Nearby, Conversation, Scene, Long-range, Unknown.
Explicit Alternative Feasible?: Yes, No, Unsure.
"""

prompt = """
---
# Annotation Prompt

You are acting as an independent linguistic annotator.

Follow the EXP010 Semantic Error Taxonomy exactly.

For every quote output exactly:

**Quote ID:** [Insert ID here]
Primary Category:
Secondary Category:
Evidence:
Context Window Needed:
Confidence:
Explicit Alternative Feasible?:
Notes:

Do not skip quotes.
"""

version_a_cols = [
    "Quote_ID", "Failure_Category", "Gold_Speaker", "Candidate_List",
    "Quote", "Context (Referring Expression)", "Explicit Signals Present?",
    "Previous Speaker", "Conversation Turn Index", "Discourse Context Length"
]

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"# EXP010 Missing 40 Quotes (BLIND)\n\n")
    f.write(f"Total Missing Quotes: {missing_count}\n\n")
    f.write(manual_summary + "\n")
    
    for idx, row in missing_df.iterrows():
        f.write(f"## Quote\n")
        for col in version_a_cols:
            f.write(f"**{col}:** {row[col]}\n")
        f.write("\n")
        
    f.write(prompt)

print(f"Generated {output_file}")
