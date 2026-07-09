import re
with open('scripts/run_exp014d_quote_aligned_syntax.py', 'r') as f:
    text = f.read()

# Change is_pronoun and add is_named
old_code = """                is_pronoun = best_nsubj.text.lower() in pronouns"""
new_code = """                is_pronoun = best_nsubj.text.lower() in pronouns
                is_named = any(t.pos_ == "PROPN" or t.text[0].isupper() for t in speaker_tokens) and not is_pronoun"""
text = text.replace(old_code, new_code)

old_code2 = """                if not is_pronoun:
                    stats["named_extracted"] += 1"""
new_code2 = """                if is_named:
                    stats["named_extracted"] += 1"""
text = text.replace(old_code2, new_code2)

old_code3 = """                    if is_correct:
                        stats["named_correct"] += 1
                    else:
                        print(f"[ERROR] Named Quote {quote_id} | Gold: {row['gold_candidate']} | Extracted Named: {speaker_text}")
                else:"""
new_code3 = """                    if is_correct:
                        stats["named_correct"] += 1
                    else:
                        print(f"[ERROR] Named Quote {quote_id} | Gold: {row['gold_candidate']} | Extracted Named: {speaker_text}")
                elif is_pronoun:"""
text = text.replace(old_code3, new_code3)

with open('scripts/run_exp014d_quote_aligned_syntax.py', 'w') as f:
    f.write(text)
