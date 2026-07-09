import pandas as pd
import os
import math

def generate_batches():
    csv_path = "results/EXP010/annotation_worksheet.csv"
    output_dir = "annotations"
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    batch_size = 25
    num_batches = math.ceil(len(df) / batch_size)
    
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
    
    for i in range(num_batches):
        batch_df = df.iloc[i*batch_size : (i+1)*batch_size]
        batch_num = i + 1
        
        # VERSION A (Blind)
        with open(os.path.join(output_dir, f"batch_{batch_num:03d}_blind.md"), "w", encoding="utf-8") as f:
            f.write(f"# EXP010 Annotation Batch {batch_num:03d} (BLIND)\n\n")
            f.write(f"Quotes: {i*batch_size + 1}-{min((i+1)*batch_size, len(df))}\n\n")
            f.write(manual_summary + "\n")
            
            for idx, row in batch_df.iterrows():
                f.write(f"## Quote {idx + 1}\n")
                for col in version_a_cols:
                    f.write(f"**{col}:** {row[col]}\n")
                f.write("\n")
                
            f.write(prompt)
            
        # VERSION B (Review)
        with open(os.path.join(output_dir, f"batch_{batch_num:03d}_review.md"), "w", encoding="utf-8") as f:
            f.write(f"# EXP010 Annotation Batch {batch_num:03d} (REVIEW)\n\n")
            f.write(f"Quotes: {i*batch_size + 1}-{min((i+1)*batch_size, len(df))}\n\n")
            
            for idx, row in batch_df.iterrows():
                f.write(f"## Quote {idx + 1}\n")
                for col in version_a_cols:
                    f.write(f"**{col}:** {row[col]}\n")
                f.write(f"**LR_Prediction:** {row['LR_Prediction']}\n")
                f.write(f"**GBM_Prediction:** {row['GBM_Prediction']}\n")
                f.write("\n")
                
    print(f"Generated {num_batches} batches in {output_dir}/")

if __name__ == "__main__":
    generate_batches()
