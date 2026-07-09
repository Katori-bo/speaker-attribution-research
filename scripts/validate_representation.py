import os
import json
import pandas as pd
import ast
from src.coreference.parser import BookNLPParser
from src.coreference.validation import RepresentationValidator
from src.coreference.alignment import AlignmentLayer

def generate_statistics():
    novel = "PrideAndPrejudice"
    out_dir = "data/raw/pdnc/booknlp_out"
    pdnc_quotes = f"data/raw/pdnc/data/{novel}/quotation_info.csv"
    
    parser = BookNLPParser()
    entities = parser.parse_entities(os.path.join(out_dir, f"{novel}.entities"))
    
    # Validations
    errors = RepresentationValidator.run_all(entities)
    print("=== Validation Errors ===")
    if errors:
        for e in errors:
            print(e)
    else:
        print("None. Coreference representation is internally valid.")
    
    # Statistics
    num_chains = len(entities)
    mentions_per_chain = [len(e.mentions) for e in entities.values()]
    singletons = sum(1 for m in mentions_per_chain if m == 1)
    avg_chain_len = sum(mentions_per_chain) / num_chains if num_chains > 0 else 0
    total_mentions = sum(mentions_per_chain)
    
    print("\n=== Representation Statistics ===")
    print(f"Number of chains: {num_chains}")
    print(f"Total mentions: {total_mentions}")
    print(f"Singleton chains: {singletons}")
    print(f"Average chain length: {avg_chain_len:.2f}")
    print(f"Invalid chains: {len(set(e.split(':')[0] for e in errors if 'Chain' in e))}")
    
    # Alignment
    print("\n=== Alignment Statistics ===")
    tokens_df = pd.read_csv(os.path.join(out_dir, f"{novel}.tokens"), sep='\t')
    alignment_layer = AlignmentLayer(tokens_df)
    
    pdnc_df = pd.read_csv(pdnc_quotes)
    
    successful_alignments = 0
    alignment_failures = 0
    
    for idx, row in pdnc_df.iterrows():
        # byte spans are stored as string representation of lists
        spans_str = row['quoteByteSpans']
        if pd.isna(spans_str):
            continue
        try:
            spans = ast.literal_eval(spans_str)
            # handle depth differences (e.g. [[395, 414], [447, 499]] or [395, 414])
            if len(spans) > 0 and isinstance(spans[0], int):
                spans = [spans]
            
            token_ids = alignment_layer.map_quote_byte_spans_to_tokens(spans)
            if len(token_ids) > 0:
                successful_alignments += 1
            else:
                alignment_failures += 1
        except Exception as e:
            alignment_failures += 1

    total_quotes = successful_alignments + alignment_failures
    align_rate = (successful_alignments / total_quotes * 100) if total_quotes > 0 else 0
    
    print(f"PDNC Quotes processed: {total_quotes}")
    print(f"Successful alignments (>= 1 token mapped): {successful_alignments}")
    print(f"Alignment failures: {alignment_failures}")
    print(f"Alignment success rate: {align_rate:.2f}%")

if __name__ == "__main__":
    generate_statistics()
