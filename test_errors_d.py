import re
with open('scripts/run_exp014d_quote_aligned_syntax.py', 'r') as f:
    text = f.read()

text = text.replace('stats["named_correct"] += 1', 'stats["named_correct"] += 1\n                    else:\n                        print(f"[ERROR] Named Quote {quote_id} | Gold: {row[\'gold_candidate\']} | Extracted Named: {speaker_text}")')
with open('scripts/run_exp014d_quote_aligned_syntax.py', 'w') as f:
    f.write(text)
