import os
import time
import pandas as pd
import json

def run_booknlp_on_sample():
    try:
        from booknlp.booknlp import BookNLP
    except ImportError:
        print("BookNLP is not installed yet.")
        return False

    model_params = {
        "pipeline": "entity,quote,coref",
        "model": "small"
    }
    
    # We will use small to ensure it runs quickly for the audit
    booknlp = BookNLP("en", model_params)
    
    novel = "PrideAndPrejudice"
    input_file = f"data/raw/pdnc/data/{novel}/novel_text.txt"
    out_dir = "data/raw/pdnc/booknlp_out"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Running BookNLP on {novel}...")
    start = time.time()
    booknlp.process(input_file, out_dir, novel)
    end = time.time()
    
    print(f"Finished {novel} in {end - start:.2f} seconds.")
    return True

def analyze_audit():
    novel = "PrideAndPrejudice"
    out_dir = "data/raw/pdnc/booknlp_out"
    tokens_file = os.path.join(out_dir, f"{novel}.tokens")
    entities_file = os.path.join(out_dir, f"{novel}.entities")
    
    if not os.path.exists(tokens_file):
        print("Outputs not found.")
        return
        
    tokens = pd.read_csv(tokens_file, sep='\t')
    entities = pd.read_csv(entities_file, sep='\t')
    
    total_pronouns = tokens[tokens['POS'] == 'PRP'].shape[0]
    # Check how many of these pronouns are part of an entity in entities
    # A bit tricky without direct token mapping, but BookNLP entities have start_token and end_token
    # BookNLP provides COREF id.
    
    print(f"Total Pronouns: {total_pronouns}")
    print(f"Total Entities (Coreference Chains mapped): {entities['COREF'].nunique()}")

if __name__ == "__main__":
    if run_booknlp_on_sample():
        analyze_audit()
