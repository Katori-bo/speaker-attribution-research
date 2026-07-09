import os
import re
import pandas as pd
from pathlib import Path

def analyze_feasibility():
    print("Loading data...")
    oracle_file = Path("results/EXP013/prediction_comparison.csv")
    if not oracle_file.exists():
        print(f"Error: {oracle_file} not found.")
        return
        
    oracle_df = pd.read_csv(oracle_file)
    failures_df = oracle_df[oracle_df['baseline_rank'] > 1].copy()
    
    features_df = pd.read_csv("data/raw/pdnc/phase2/candidate_features.csv")
    quote_bounds = features_df.groupby('quote_id').first().reset_index()
    failures_df = failures_df.merge(quote_bounds[['quote_id', 'quote_start_byte', 'quote_end_byte']], on='quote_id', how='left')
    
    candidates_per_quote = features_df.groupby('quote_id')['candidate'].apply(set).to_dict()
    
    print(f"Total EXP012B failures found: {len(failures_df)}")
    
    verbs = r"(said|asked|cried|replied|answered|exclaimed|whispered|shouted|added|continued|observed|remarked|murmured)"
    pronouns = r"(he|she|they|I|we|you)"
    name = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
    nominal = r"(the\s+[a-z]+(?:\s+[a-z]+)?)"
    
    pattern1 = re.compile(rf"\b(?:{name}|{pronouns}|{nominal})\s+{verbs}\b", re.IGNORECASE)
    pattern2 = re.compile(rf"\b{verbs}\s+(?:{name}|{pronouns}|{nominal})\b", re.IGNORECASE)
    
    stats = {
        "total": len(failures_df),
        "tag_exists": 0,
        "speaker_mention_detected": 0,
        "mention_maps_to_entity": 0,
        "candidate_exists": 0
    }
    
    for _, row in failures_df.iterrows():
        novel = row['novel']
        novel_txt_path = Path(f"data/raw/pdnc/data/{novel}/{novel}.txt")
        if not novel_txt_path.exists():
            txt_files = list(Path(f"data/raw/pdnc/data/{novel}").glob("*.txt"))
            if txt_files:
                novel_txt_path = txt_files[0]
                
        if not novel_txt_path.exists():
            continue
            
        with open(novel_txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            start = int(row['quote_start_byte'])
            end = int(row['quote_end_byte'])
            
            ctx_start = max(0, start - 150)
            ctx_end = min(len(content), end + 150)
            context = content[ctx_start:start] + " " + content[end:ctx_end]
            context = context.replace('\n', ' ')
            
            tag_found = False
            speaker_mention = None
            
            m1 = pattern1.search(context)
            if m1:
                tag_found = True
                speaker_mention = next((g for g in m1.groups()[:-1] if g is not None), None)
            
            if not tag_found:
                m2 = pattern2.search(context)
                if m2:
                    tag_found = True
                    speaker_mention = next((g for g in m2.groups()[1:] if g is not None), None)
            
            if tag_found:
                stats["tag_exists"] += 1
                if speaker_mention:
                    stats["speaker_mention_detected"] += 1
                    speaker_mention = speaker_mention.strip()
                    
                    is_pronoun = bool(re.match(rf"^{pronouns}$", speaker_mention, re.IGNORECASE))
                    quote_id = row['quote_id']
                    cands = candidates_per_quote.get(quote_id, set())
                    
                    maps_to_entity = False
                    if is_pronoun:
                        maps_to_entity = True 
                    else:
                        for cand in cands:
                            if speaker_mention.lower() in cand.lower() or cand.lower() in speaker_mention.lower():
                                maps_to_entity = True
                                break
                                
                    if maps_to_entity:
                        stats["mention_maps_to_entity"] += 1
                        stats["candidate_exists"] += 1

    print("\n=== EXP014B.0 Attribution Feasibility Analysis ===")
    print(f"Residual errors: {stats['total']}")
    print(f"Attribution tags: {stats['tag_exists']}")
    print(f"Speaker mention detected: {stats['speaker_mention_detected']}")
    print(f"Mention maps to entity: {stats['mention_maps_to_entity']}")
    print(f"Candidate exists: {stats['candidate_exists']}")
    
    if stats['total'] > 0:
        recovery = (stats['candidate_exists'] / stats['total']) * 100
        print(f"Upper bound recovery: {recovery:.1f}%")

if __name__ == "__main__":
    analyze_feasibility()
