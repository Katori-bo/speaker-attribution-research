import csv
import os
import ast

def analyze_addressee_coverage():
    base_dir = "data/raw/pdnc/data"
    all_books = os.listdir(base_dir)
    
    total_quotes = 0
    quotes_with_addressee = 0
    ambiguous_cases = 0
    failure_examples = []
    
    for book in all_books:
        quote_file = os.path.join(base_dir, book, "quotation_info.csv")
        if not os.path.exists(quote_file):
            continue
            
        with open(quote_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_quotes += 1
                
                addressees_raw = row.get('addressees', '')
                
                if not addressees_raw or addressees_raw == 'nan' or addressees_raw == '':
                    continue
                    
                try:
                    addressees = ast.literal_eval(addressees_raw)
                    if isinstance(addressees, list) and len(addressees) > 0:
                        quotes_with_addressee += 1
                        
                        if len(addressees) > 1:
                            ambiguous_cases += 1
                            
                except (SyntaxError, ValueError):
                    if len(addressees_raw.strip()) > 0:
                        quotes_with_addressee += 1
                    else:
                        if len(failure_examples) < 5:
                            failure_examples.append(f"Failed to parse: {addressees_raw}")

    coverage = (quotes_with_addressee / total_quotes) * 100 if total_quotes > 0 else 0
    
    print(f"Total quotes analyzed (PDNC Gold): {total_quotes}")
    print(f"Quotes with extractable addressee (PDNC Gold): {quotes_with_addressee}")
    print(f"Coverage %: {coverage:.2f}%")
    print(f"Extraction source: PDNC quotation_info.csv (Gold annotations)")
    print(f"Ambiguous cases (Multiple addressees): {ambiguous_cases}")
    print("Failure examples:")
    for f in failure_examples:
        print(f" - {f}")

if __name__ == "__main__":
    analyze_addressee_coverage()
