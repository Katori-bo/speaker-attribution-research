import os
import re
import pandas as pd
from pathlib import Path

def run_structured_oracle():
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
    
    # 1. Post Quote Patterns: ^[\s\W]* ensures it's anchored to the start of the right context
    post_pattern_vs = re.compile(rf"^[\s\W]*{verbs}\s+(?:{name}|{pronouns}|{nominal})\b", re.IGNORECASE)
    post_pattern_sv = re.compile(rf"^[\s\W]*(?:{name}|{pronouns}|{nominal})\s+{verbs}\b", re.IGNORECASE)
    
    # 2. Pre Quote Patterns: [\s\W]*$ ensures it's anchored to the end of the left context
    pre_pattern_sv = re.compile(rf"\b(?:{name}|{pronouns}|{nominal})\s+{verbs}[\s\W]*$", re.IGNORECASE)
    pre_pattern_vs = re.compile(rf"\b{verbs}\s+(?:{name}|{pronouns}|{nominal})[\s\W]*$", re.IGNORECASE)
    
    resolvable_count = 0
    correct_matches = 0
    total_tags = 0
    
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
            
            # 80 chars boundary
            left_context = content[max(0, start - 80):start].replace('\n', ' ')
            right_context = content[end:min(len(content), end + 80)].replace('\n', ' ')
            
            tag_found = False
            speaker_mention = None
            
            # Post-quote is generally more common and reliable for trailing attributions
            m_post_vs = post_pattern_vs.search(right_context)
            if m_post_vs:
                tag_found = True
                speaker_mention = next((g for g in m_post_vs.groups()[1:] if g is not None), None)
            
            if not tag_found:
                m_post_sv = post_pattern_sv.search(right_context)
                if m_post_sv:
                    tag_found = True
                    speaker_mention = next((g for g in m_post_sv.groups()[:-1] if g is not None), None)
            
            # Pre-quote
            if not tag_found:
                m_pre_sv = pre_pattern_sv.search(left_context)
                if m_pre_sv:
                    tag_found = True
                    speaker_mention = next((g for g in m_pre_sv.groups()[:-1] if g is not None), None)
            
            if not tag_found:
                m_pre_vs = pre_pattern_vs.search(left_context)
                if m_pre_vs:
                    tag_found = True
                    speaker_mention = next((g for g in m_pre_vs.groups()[1:] if g is not None), None)
            
            if tag_found and speaker_mention:
                total_tags += 1
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
                        mention_lower = speaker_mention.lower()
                        if mention_lower in gold or gold in mention_lower:
                            is_correct = True
                        elif matched_candidate and (matched_candidate.lower() in gold or gold in matched_candidate.lower()):
                            is_correct = True
                    else:
                        p = speaker_mention.lower()
                        male_pronouns = ['he']
                        female_pronouns = ['she']
                        if p in male_pronouns and not ('mrs.' in gold or 'miss' in gold or 'lady' in gold or 'mary' in gold or 'elizabeth' in gold or 'jane' in gold):
                            is_correct = True
                        elif p in female_pronouns and ('mrs.' in gold or 'miss' in gold or 'lady' in gold or 'mary' in gold or 'elizabeth' in gold or 'jane' in gold):
                            is_correct = True
                        elif p in ['i', 'we', 'you', 'they']:
                            is_correct = True 

                    if is_correct:
                        correct_matches += 1
                    else:
                        print(f"[ERROR] Quote {quote_id} | Gold: {row['gold_candidate']} | Extracted: {speaker_mention}")

    print("\n=== EXP014B.0c Structured Attribution Oracle Validation ===")
    print(f"Total EXP012B failures: {len(failures_df)}")
    print(f"Extracted tags: {total_tags}")
    print(f"Resolvable attribution tags: {resolvable_count}")
    print(f"Correct speaker matches: {correct_matches}")
    
    if resolvable_count > 0:
        precision = (correct_matches / resolvable_count) * 100
        print(f"Precision: {precision:.1f}%")

if __name__ == "__main__":
    run_structured_oracle()
