import os
import re
import pandas as pd
import spacy
from pathlib import Path

def run_signal_audit():
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
    # Use ALL quotes this time
    
    features_df = pd.read_csv("data/raw/pdnc/phase2/candidate_features.csv")
    quote_bounds = features_df.groupby('quote_id').first().reset_index()
    oracle_df = oracle_df.merge(quote_bounds[['quote_id', 'quote_start_byte', 'quote_end_byte']], on='quote_id', how='left')
    
    candidates_per_quote = features_df.groupby('quote_id')['candidate'].apply(set).to_dict()
    
    speech_verbs = {"say", "ask", "reply", "cry", "answer", "exclaim", "whisper", "shout", "add", "continue", "observe", "remark", "murmur", "think"}
    pronouns = {"he", "she", "they", "i", "we", "you"}
    
    stats = {
        "total": len(oracle_df),
        "named_extracted": 0,
        "named_correct": 0,
        "successes_covered": 0,
        "failures_covered": 0,
    }
    
    for _, row in oracle_df.iterrows():
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
            
            left_context = content[max(0, start - 150):start]
            right_context = content[end:min(len(content), end + 150)]
            
            # Region 1: Right attached tag
            match_right = re.search(r'[\.\!\?\n]', right_context)
            if match_right:
                right_span = right_context[:match_right.end()]
            else:
                right_span = right_context
            
            # Region 2: Left attached tag
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
                        for child in token.children:
                            if child.dep_ in ("nsubj", "nsubjpass"):
                                best_nsubj = child
                                best_verb = token
                                break
                        if best_nsubj:
                            break
            
            # Parse Region 2 if Region 1 failed
            if not best_nsubj and left_span:
                doc_left = nlp(left_span)
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
                is_pronoun = best_nsubj.text.lower() in pronouns
                if is_pronoun:
                    continue # Ignore pronouns entirely per instructions
                    
                speaker_tokens = [best_nsubj]
                for child in best_nsubj.children:
                    if child.dep_ in ("compound", "det", "amod"):
                        speaker_tokens.append(child)
                speaker_tokens = sorted(speaker_tokens, key=lambda x: x.i)
                
                is_named = any(t.pos_ == "PROPN" or t.text[0].isupper() for t in speaker_tokens)
                if not is_named:
                    continue
                    
                speaker_text = " ".join([t.text for t in speaker_tokens])
                
                stats["named_extracted"] += 1
                
                quote_id = row['quote_id']
                cands = candidates_per_quote.get(quote_id, set())
                gold = str(row['gold_candidate']).lower()
                
                matched_candidate = None
                for cand in cands:
                    if best_nsubj.text.lower() in cand.lower() or cand.lower() in best_nsubj.text.lower():
                        matched_candidate = cand
                        break
                        
                is_correct = False
                mention_lower = speaker_text.lower()
                if mention_lower in gold or gold in mention_lower:
                    is_correct = True
                elif matched_candidate and (matched_candidate.lower() in gold or gold in matched_candidate.lower()):
                    is_correct = True
                    
                if is_correct:
                    stats["named_correct"] += 1
                    
                if row['baseline_rank'] == 1.0:
                    stats["successes_covered"] += 1
                else:
                    stats["failures_covered"] += 1

    print("\n=== EXP014D.1 Dataset-level Signal Audit ===")
    print(f"Total examples: {stats['total']}")
    print(f"Named attribution: {stats['named_extracted']}")
    coverage_pct = (stats['named_extracted'] / stats['total']) * 100
    print(f"Coverage: {coverage_pct:.1f}%")
    
    precision = 0
    if stats['named_extracted'] > 0:
        precision = (stats['named_correct'] / stats['named_extracted']) * 100
    print(f"Precision: {precision:.1f}%")
    
    print(f"EXP012 failures covered: {stats['failures_covered']}")
    print(f"EXP012 successes covered: {stats['successes_covered']}")
    
    if stats['failures_covered'] >= 30:
        expected_role = "recovery"
    elif stats['named_extracted'] >= 300 and precision >= 95:
        expected_role = "calibration"
    else:
        expected_role = "insufficient"
        
    print(f"Expected role: {expected_role}")

if __name__ == "__main__":
    run_signal_audit()
