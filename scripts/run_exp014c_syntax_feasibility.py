import os
import re
import pandas as pd
import spacy
from pathlib import Path

def run_syntax_feasibility():
    print("Loading data and model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
        
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
    
    speech_verbs = {"say", "ask", "reply", "cry", "answer", "exclaim", "whisper", "shout", "add", "continue", "observe", "remark", "murmur", "think"}
    pronouns = {"he", "she", "they", "i", "we", "you"}
    
    stats = {
        "total": len(failures_df),
        "found": 0,
        "named_extracted": 0,
        "named_correct": 0,
        "pronoun_extracted": 0,
        "pronoun_correct": 0,
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
            
            # Extract a window that includes the quote so parser has context
            ctx_start = max(0, start - 150)
            ctx_end = min(len(content), end + 150)
            context = content[ctx_start:ctx_end].replace('\n', ' ')
            
            # The quote starts at (start - ctx_start) and ends at (end - ctx_start) in the context string
            local_start = start - ctx_start
            local_end = end - ctx_start
            
            doc = nlp(context)
            
            best_verb = None
            min_dist = float('inf')
            best_nsubj = None
            
            for token in doc:
                if token.lemma_.lower() in speech_verbs:
                    # Make sure the verb is outside the quote
                    if token.idx >= local_start and token.idx < local_end:
                        continue
                        
                    # Find nsubj
                    nsubj = None
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            nsubj = child
                            break
                            
                    if nsubj is not None:
                        # Distance to quote
                        dist = min(abs(token.idx - local_start), abs(token.idx - local_end))
                        if dist < min_dist:
                            min_dist = dist
                            best_verb = token
                            best_nsubj = nsubj
                            
            if best_nsubj is not None:
                stats["found"] += 1
                
                # Expand nsubj to include compounds for full name, e.g., "Mr. Darcy"
                speaker_tokens = [best_nsubj]
                for child in best_nsubj.children:
                    if child.dep_ in ("compound", "det", "amod"): # things like "Mr.", "the", "old"
                        speaker_tokens.append(child)
                speaker_tokens = sorted(speaker_tokens, key=lambda x: x.i)
                speaker_text = " ".join([t.text for t in speaker_tokens])
                
                is_pronoun = best_nsubj.text.lower() in pronouns
                
                quote_id = row['quote_id']
                cands = candidates_per_quote.get(quote_id, set())
                gold = str(row['gold_candidate']).lower()
                
                matched_candidate = None
                if is_pronoun:
                    matched_candidate = speaker_text
                else:
                    for cand in cands:
                        if best_nsubj.text.lower() in cand.lower() or cand.lower() in best_nsubj.text.lower():
                            matched_candidate = cand
                            break
                
                is_correct = False
                if not is_pronoun:
                    stats["named_extracted"] += 1
                    mention_lower = speaker_text.lower()
                    if mention_lower in gold or gold in mention_lower:
                        is_correct = True
                    elif matched_candidate and (matched_candidate.lower() in gold or gold in matched_candidate.lower()):
                        is_correct = True
                        
                    if is_correct:
                        stats["named_correct"] += 1
                    else:
                        print(f"[ERROR] Named Quote {quote_id} | Gold: {row['gold_candidate']} | Extracted Named: {speaker_text}")
                else:
                    stats["pronoun_extracted"] += 1
                    p = best_nsubj.text.lower()
                    male_pronouns = ['he']
                    female_pronouns = ['she']
                    if p in male_pronouns and not ('mrs.' in gold or 'miss' in gold or 'lady' in gold or 'mary' in gold or 'elizabeth' in gold or 'jane' in gold):
                        is_correct = True
                    elif p in female_pronouns and ('mrs.' in gold or 'miss' in gold or 'lady' in gold or 'mary' in gold or 'elizabeth' in gold or 'jane' in gold):
                        is_correct = True
                    elif p in ['i', 'we', 'you', 'they']:
                        is_correct = True
                        
                    if is_correct:
                        stats["pronoun_correct"] += 1

    print("\n=== EXP014C.0 Syntax Attribution Feasibility ===")
    print(f"Total residual errors: {stats['total']}")
    print(f"Syntax attribution found: {stats['found']}")
    print("")
    
    named_prec = 0
    if stats['named_extracted'] > 0:
        named_prec = (stats['named_correct'] / stats['named_extracted']) * 100
    print("Named:")
    print(f" count: {stats['named_extracted']}")
    print(f" precision: {named_prec:.1f}%")
    print("")
    
    pronoun_prec = 0
    if stats['pronoun_extracted'] > 0:
        pronoun_prec = (stats['pronoun_correct'] / stats['pronoun_extracted']) * 100
    print("Pronoun:")
    print(f" count: {stats['pronoun_extracted']}")
    print(f" precision: {pronoun_prec:.1f}%")
    print("")
    
    overall_recovery = 0
    if stats['total'] > 0:
        overall_recovery = ((stats['named_correct'] + stats['pronoun_correct']) / stats['total']) * 100
    print(f"Overall oracle recovery: {overall_recovery:.1f}%")

if __name__ == "__main__":
    run_syntax_feasibility()
