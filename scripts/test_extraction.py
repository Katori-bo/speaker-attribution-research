import os
import csv

def test_extraction(book_name="PrideAndPrejudice"):
    booknlp_dir = "data/raw/pdnc/booknlp_out"
    tokens_file = os.path.join(booknlp_dir, f"{book_name}.tokens")
    quotes_file = os.path.join(booknlp_dir, f"{book_name}.quotes")
    
    if not os.path.exists(tokens_file) or not os.path.exists(quotes_file):
        print("Files not found.")
        return

    tokens = []
    with open(tokens_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            tokens.append(row)
            
    quotes = []
    with open(quotes_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            quotes.append(row)
            
    doc_tokens = {int(t['token_ID_within_document']): t for t in tokens}
    
    extracted_count = 0
    ambiguous_cases = 0
    total_quotes = len(quotes)
    
    failure_examples = []
    
    for quote in quotes:
        m_start = int(quote['mention_start'])
        m_end = int(quote['mention_end'])
        
        has_dep_addressee = False
        has_voc = False
        
        if m_start >= 0 and m_end >= 0:
            head_verb_id = -1
            
            for t_id in range(m_start, m_end + 1):
                if t_id in doc_tokens:
                    tok = doc_tokens[t_id]
                    # syntactic_head_ID is the token_ID_within_document of the head!
                    head_doc_id = int(tok['syntactic_head_ID'])
                    if head_doc_id != t_id and head_doc_id in doc_tokens:
                        if not (m_start <= head_doc_id <= m_end):
                            head_verb_id = head_doc_id
                            break
                            
            if head_verb_id != -1:
                verb_tok = doc_tokens[head_verb_id]
                sent_id = verb_tok['sentence_ID']
                
                addressees_found = []
                
                for t_id, tok in doc_tokens.items():
                    # check if the token is a child of the speech verb
                    if int(tok['syntactic_head_ID']) == head_verb_id:
                        if tok['dependency_relation'] == 'npadvmod':
                            has_voc = True
                            addressees_found.append(tok['word'])
                            
                        if tok['dependency_relation'] == 'prep' and tok['word'].lower() == 'to':
                            # find pobj of this prep
                            for t2_id, tok2 in doc_tokens.items():
                                if int(tok2['syntactic_head_ID']) == t_id and tok2['dependency_relation'] == 'pobj':
                                    has_dep_addressee = True
                                    addressees_found.append(tok2['word'])
                                            
                if len(addressees_found) > 0:
                    extracted_count += 1
                    if len(addressees_found) > 1:
                        ambiguous_cases += 1
                else:
                    if len(failure_examples) < 5:
                        failure_examples.append(f"Quote: {quote['quote']} (Speaker: {quote['mention_phrase']})")
                        
    coverage = (extracted_count / total_quotes) * 100 if total_quotes > 0 else 0
    
    print(f"Total quotes analyzed: {total_quotes}")
    print(f"Quotes with extractable addressee: {extracted_count}")
    print(f"Coverage %: {coverage:.2f}%")
    print(f"Extraction source: Dependency parsing (npadvmod & prep->pobj)")
    print(f"Ambiguous cases: {ambiguous_cases}")
    print("Failure examples:")
    for f in failure_examples:
        print(f" - {f}")

if __name__ == "__main__":
    test_extraction()
