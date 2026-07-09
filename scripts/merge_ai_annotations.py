import os
import re
import pandas as pd
import glob

def parse_annotation_files(directory):
    files = glob.glob(os.path.join(directory, "*"))
    
    parsed_quotes = {}
    
    for file in files:
        if os.path.isdir(file): continue
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Split by Quote ID or Quote_ID
        blocks = re.split(r'\*?\*?Quote[_ ]ID:\*?\*?', content, flags=re.IGNORECASE)
        
        for block in blocks[1:]:
            lines = [line.strip() for line in block.split('\n')]
            quote_id = lines[0].strip().replace('*', '').replace('#', '')
            
            data = {"Quote_ID": quote_id}
            
            # Find fields
            current_field = None
            for original_line in lines[1:]:
                line = original_line.replace('*', '').replace('#', '').strip()
                if line.startswith("Primary Category:"):
                    current_field = "Primary Category"
                    data[current_field] = line.split(":", 1)[1].strip()
                elif line.startswith("Secondary Category:"):
                    current_field = "Secondary Category"
                    data[current_field] = line.split(":", 1)[1].strip()
                elif line.startswith("Evidence:"):
                    current_field = "Evidence"
                    data[current_field] = line.split(":", 1)[1].strip()
                elif line.startswith("Context Window Needed:"):
                    current_field = "Context Window Needed"
                    data[current_field] = line.split(":", 1)[1].strip()
                elif line.startswith("Confidence:"):
                    current_field = "Confidence"
                    data[current_field] = line.split(":", 1)[1].strip()
                elif line.startswith("Explicit Alternative Feasible?:"):
                    current_field = "Explicit Alternative Feasible?"
                    data[current_field] = line.split("?:", 1)[1].strip()
                elif line.startswith("Notes:"):
                    current_field = "Notes"
                    data[current_field] = line.split(":", 1)[1].strip()
                elif current_field:
                    if line:
                        data[current_field] += " " + line
            
            parsed_quotes[quote_id] = data
            
    return parsed_quotes

def merge_annotations():
    parsed_main = parse_annotation_files("/home/Aditya/Documents/research")
    parsed_left = parse_annotation_files("/home/Aditya/Documents/research/left folder")
    parsed_last = parse_annotation_files("/home/Aditya/Documents/research/last")
    
    parsed = {**parsed_main, **parsed_left, **parsed_last}
    
    # Read the original worksheet
    orig_df = pd.read_csv("results/EXP010/annotation_worksheet.csv")
    
    annotation_cols = [
        'ANNOTATION: Primary Category', 'ANNOTATION: Secondary Category',
        'ANNOTATION: Evidence', 'ANNOTATION: Context Window Needed',
        'ANNOTATION: Confidence', 'ANNOTATION: Explicit Alternative Feasible?',
        'ANNOTATION: Notes'
    ]
    for col in annotation_cols:
        orig_df[col] = orig_df[col].astype(object)
        
    for idx, row in orig_df.iterrows():
        qid = row['Quote_ID']
        if qid in parsed:
            p = parsed[qid]
            orig_df.at[idx, 'ANNOTATION: Primary Category'] = p.get('Primary Category', '')
            orig_df.at[idx, 'ANNOTATION: Secondary Category'] = p.get('Secondary Category', '')
            orig_df.at[idx, 'ANNOTATION: Evidence'] = p.get('Evidence', '')
            orig_df.at[idx, 'ANNOTATION: Context Window Needed'] = p.get('Context Window Needed', '')
            orig_df.at[idx, 'ANNOTATION: Confidence'] = p.get('Confidence', '')
            orig_df.at[idx, 'ANNOTATION: Explicit Alternative Feasible?'] = p.get('Explicit Alternative Feasible?', '')
            orig_df.at[idx, 'ANNOTATION: Notes'] = p.get('Notes', '')
            
    orig_df.to_csv("results/EXP010/semantic_annotations_master.csv", index=False)
    print(f"Successfully parsed {len(parsed)} annotations and merged into results/EXP010/semantic_annotations_master.csv")
    
if __name__ == "__main__":
    merge_annotations()
