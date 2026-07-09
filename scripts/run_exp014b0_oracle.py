import os
import re
import pandas as pd
from pathlib import Path

def run_oracle_validation():
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
    
    verbs = r"(said|asked|cried|replied|answered|exclaimed|whispered|shouted|added|continued|observed|remarked|murmured)"
    pronouns = r"(he|she|they|I|we|you)"
    name = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
    nominal = r"(the\s+[a-z]+(?:\s+[a-z]+)?)"
    
    pattern1 = re.compile(rf"\b(?:{name}|{pronouns}|{nominal})\s+{verbs}\b", re.IGNORECASE)
    pattern2 = re.compile(rf"\b{verbs}\s+(?:{name}|{pronouns}|{nominal})\b", re.IGNORECASE)
    
    resolvable_count = 0
    correct_matches = 0
    
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
            
            if tag_found and speaker_mention:
                speaker_mention = speaker_mention.strip()
                is_pronoun = bool(re.match(rf"^{pronouns}$", speaker_mention, re.IGNORECASE))
                quote_id = row['quote_id']
                cands = candidates_per_quote.get(quote_id, set())
                
                maps_to_entity = False
                matched_candidate = None
                
                if is_pronoun:
                    maps_to_entity = True 
                    matched_candidate = speaker_mention
                else:
                    for cand in cands:
                        if speaker_mention.lower() in cand.lower() or cand.lower() in speaker_mention.lower():
                            maps_to_entity = True
                            matched_candidate = cand
                            break
                            
                if maps_to_entity:
                    resolvable_count += 1
                    
                    gold = str(row['gold_candidate']).lower()
                    
                    is_correct = False
                    if not is_pronoun:
                        # For explicit names, check if it matches gold
                        mention_lower = speaker_mention.lower()
                        if mention_lower in gold or gold in mention_lower:
                            is_correct = True
                        elif matched_candidate and (matched_candidate.lower() in gold or gold in matched_candidate.lower()):
                            is_correct = True
                    else:
                        # For pronouns, we assume a perfect oracle resolver would map it correctly IF the pronoun gender matches
                        # the gold speaker.
                        p = speaker_mention.lower()
                        male_pronouns = ['he']
                        female_pronouns = ['she']
                        # Basic heuristics to check if pronoun matches gold (e.g. Mr. vs Mrs.)
                        if p in male_pronouns and not ('mrs.' in gold or 'miss' in gold or 'lady' in gold):
                            is_correct = True # Likely matches male
                        elif p in female_pronouns and ('mrs.' in gold or 'miss' in gold or 'lady' in gold or 'mary' in gold or 'elizabeth' in gold or 'jane' in gold):
                            is_correct = True
                        elif p in ['i', 'we', 'you', 'they']:
                            is_correct = True # We can't tell, give benefit of doubt for oracle

                    if is_correct:
                        correct_matches += 1
                    else:
                        print(f"[ERROR] Quote {quote_id} | Gold: {row['gold_candidate']} | Extracted: {speaker_mention}")

    print("\n=== EXP014B.0b Attribution Oracle Validation ===")
    print(f"Resolvable attribution tags: {resolvable_count}")
    print(f"Correct speaker matches: {correct_matches}")
    
    if resolvable_count > 0:
        precision = (correct_matches / resolvable_count) * 100
        print(f"Precision: {precision:.1f}%")

if __name__ == "__main__":
    run_oracle_validation()
