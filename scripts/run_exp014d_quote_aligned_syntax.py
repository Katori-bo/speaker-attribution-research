import os
import re
import pandas as pd
import spacy
from pathlib import Path

def run_quote_aligned_syntax():
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
            
            # Extract left and right context up to 150 chars
            left_context = content[max(0, start - 150):start]
            right_context = content[end:min(len(content), end + 150)]
            
            # Region 1: Right attached tag (quote_end -> next sentence boundary)
            # Find the first sentence boundary
            match_right = re.search(r'[\.\!\?\n]', right_context)
            if match_right:
                right_span = right_context[:match_right.end()]
            else:
                right_span = right_context
            
            # Region 2: Left attached tag (previous sentence boundary -> quote_start)
            match_left = re.search(r'[\.\!\?\n]', left_context[::-1])
            if match_left:
                left_span = left_context[-match_left.end():]
            else:
                left_span = left_context
                
            right_span = right_span.strip()
            left_span = left_span.strip()
            
            best_verb = None
            best_nsubj = None
            
            # Parse Region 1
            if right_span:
                doc_right = nlp(right_span)
                for token in doc_right:
                    if token.lemma_.lower() in speech_verbs:
                        # Find nsubj
                        for child in token.children:
                            if child.dep_ in ("nsubj", "nsubjpass"):
                                best_nsubj = child
                                best_verb = token
                                break
                        if best_nsubj:
                            break # Found first speech verb with subject in right region
            
            # Parse Region 2 if Region 1 failed
            if not best_nsubj and left_span:
                doc_left = nlp(left_span)
                # For left region, we want the *last* speech verb
                for token in reversed(doc_left):
                    if token.lemma_.lower() in speech_verbs:
                        for child in token.children:
                            if child.dep_ in ("nsubj", "nsubjpass"):
                                best_nsubj = child
                                best_verb = token
                                break
                        if best_nsubj:
                            break
                            
            if best_nsubj is not None:
                stats["found"] += 1
                
                # Expand nsubj to include compounds for full name, e.g., "Mr. Darcy"
                speaker_tokens = [best_nsubj]
                for child in best_nsubj.children:
                    if child.dep_ in ("compound", "det", "amod"):
                        speaker_tokens.append(child)
                speaker_tokens = sorted(speaker_tokens, key=lambda x: x.i)
                speaker_text = " ".join([t.text for t in speaker_tokens])
                
                is_pronoun = best_nsubj.text.lower() in pronouns
                is_named = any(t.pos_ == "PROPN" or t.text[0].isupper() for t in speaker_tokens) and not is_pronoun
                
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
                if is_named:
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
                elif is_pronoun:
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

    print("\n=== EXP014D.0 Quote-Aligned Syntax Attribution Feasibility ===")
    print(f"Total residual errors: {stats['total']}")
    print(f"Syntax attribution found: {stats['found']}")
    print("")
    
    named_prec = 0
    if stats['named_extracted'] > 0:
        named_prec = (stats['named_correct'] / stats['named_extracted']) * 100
    print("Named:")
    print(f" found: {stats['named_extracted']}")
    print(f" precision: {named_prec:.1f}%")
    print("")
    
    pronoun_prec = 0
    if stats['pronoun_extracted'] > 0:
        pronoun_prec = (stats['pronoun_correct'] / stats['pronoun_extracted']) * 100
    print("Pronoun:")
    print(f" found: {stats['pronoun_extracted']}")
    print(f" precision: {pronoun_prec:.1f}%")
    print("")

if __name__ == "__main__":
    run_quote_aligned_syntax()
