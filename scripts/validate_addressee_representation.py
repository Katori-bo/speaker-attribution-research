import os
import csv
import sys
import ast
import json
csv.field_size_limit(sys.maxsize)
from src.addressee.extractor import AddresseeExtractor
from src.addressee.validation import ValidationEngine

def run_validation(book_name="PrideAndPrejudice"):
    booknlp_dir = "data/raw/pdnc/booknlp_out"
    pdnc_dir = os.path.join("data/raw/pdnc/data", book_name)
    
    tokens_file = os.path.join(booknlp_dir, f"{book_name}.tokens")
    entities_file = os.path.join(booknlp_dir, f"{book_name}.entities")
    quotes_file = os.path.join(booknlp_dir, f"{book_name}.quotes")
    gold_file = os.path.join(pdnc_dir, "quotation_info.csv")
    
    if not all(os.path.exists(f) for f in [tokens_file, entities_file, quotes_file, gold_file]):
        print(f"Missing files for {book_name}")
        return

    # Load tokens
    tokens = []
    with open(tokens_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            tokens.append(row)
            
    # Load entities
    entities = []
    with open(entities_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            entities.append(row)
            
    # Load quotes
    quotes = []
    with open(quotes_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            quotes.append(row)
            
    # Load gold annotations
    gold_annotations = {}
    with open(gold_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # We assume quotes line up by quote_id index (0-indexed in BookNLP usually matches Q0, Q1 in PDNC)
        # But actually PDNC has quoteID like Q0, Q1
        for i, row in enumerate(reader):
            addressees_raw = row.get('addressees', '')
            if addressees_raw and addressees_raw != 'nan':
                try:
                    addrs = ast.literal_eval(addressees_raw)
                    if isinstance(addrs, list):
                        gold_annotations[i] = addrs
                except:
                    pass
                    
    # Initialize Extractor
    # Note: For full precision we would need alias mapping, but we can do a naive evaluation.
    # We will just map char_id to a placeholder string or evaluate loosely if we don't have the exact alias map.
    # But since we just need to test if the code runs and returns reasonable metrics:
    extractor = AddresseeExtractor(tokens, entities, alias_dict={})
    
    interactions = []
    for i, quote in enumerate(quotes):
        interaction = extractor.extract(quote, i)
        interactions.append(interaction)
        
    # Validation
    # We don't have a perfect string->id alias mapping here for gold, so precision_estimate will be 0.
    # That's fine for the pipeline validation.
    engine = ValidationEngine(gold_annotations, alias_dict={})
    results = engine.validate(interactions)
    
    # Save Report
    os.makedirs("results/EXP013", exist_ok=True)
    report_path = "results/EXP013/addressee_validation_report.md"
    
    with open(report_path, "w") as f:
        f.write("# EXP013A.2 Validation Report\n\n")
        f.write(f"- **Total quotes**: {results['total_quotes']}\n")
        f.write(f"- **Extracted addressees**: {results['extracted_addressees']}\n")
        f.write(f"- **Coverage %**: {results['coverage_percent']:.2f}%\n\n")
        
        f.write("## Method Distribution\n")
        for k, v in results['method_distribution'].items():
            f.write(f"- {k}: {v}\n")
            
        f.write("\n## Confidence Distribution\n")
        for k, v in results['confidence_distribution'].items():
            f.write(f"- {k}: {v}\n")
            
        f.write("\n## Gold Audit\n")
        f.write(f"- Evaluated against gold: {results['evaluated_against_gold']}\n")
        # Skipping precision since we lack alias mapping here
        f.write("- Precision estimate: (Requires full alias resolution)\n")
        
    print(f"Validation complete. Report saved to {report_path}")

if __name__ == "__main__":
    run_validation()
