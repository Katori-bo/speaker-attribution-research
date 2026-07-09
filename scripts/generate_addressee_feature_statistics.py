import os
import csv
import sys
from collections import defaultdict

# Increase csv limits
csv.field_size_limit(sys.maxsize)

from src.addressee.extractor import AddresseeExtractor
from src.addressee.state import InteractionStateUpdater
from src.addressee.features import extract_addressee_features

def run_feature_statistics(book_name="PrideAndPrejudice"):
    booknlp_dir = "data/raw/pdnc/booknlp_out"
    tokens_file = os.path.join(booknlp_dir, f"{book_name}.tokens")
    entities_file = os.path.join(booknlp_dir, f"{book_name}.entities")
    quotes_file = os.path.join(booknlp_dir, f"{book_name}.quotes")
    
    if not all(os.path.exists(f) for f in [tokens_file, entities_file, quotes_file]):
        print(f"Missing files for {book_name}")
        return

    # Load data
    tokens = []
    with open(tokens_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            tokens.append(row)
            
    entities = []
    with open(entities_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            entities.append(row)
            
    quotes = []
    with open(quotes_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            quotes.append(row)
            
    extractor = AddresseeExtractor(tokens, entities, alias_dict={})
    updater = InteractionStateUpdater(history_limit=10)
    
    feature_stats = {
        "total_evaluated": 0,
        "candidate_was_addressed_true": 0,
        "recencies": [],
        "missing_recency": 0,
        "transition_active": 0,
    }
    
    # We want to measure correlation between features as well
    # e.g., how often both candidate_was_addressed and transition_active are True
    overlap = 0
    
    # Run through quotes
    for i, quote in enumerate(quotes):
        try:
            true_speaker_id = int(quote.get('char_id', -1))
        except ValueError:
            true_speaker_id = -1
            
        if true_speaker_id != -1:
            # Evaluate features for the true speaker BEFORE updating the state with current quote
            features = extract_addressee_features(true_speaker_id, i, updater.state)
            
            feature_stats["total_evaluated"] += 1
            if features["candidate_was_addressed"]:
                feature_stats["candidate_was_addressed_true"] += 1
                
            recency = features["addressee_recency"]
            if recency != -1:
                feature_stats["recencies"].append(recency)
            else:
                feature_stats["missing_recency"] += 1
                
            if features["speaker_addressee_transition"]:
                feature_stats["transition_active"] += 1
                
            if features["candidate_was_addressed"] and features["speaker_addressee_transition"]:
                overlap += 1
        
        # Now process the quote and update state
        interaction = extractor.extract(quote, i)
        updater.update(interaction)
        
    total = feature_stats["total_evaluated"]
    if total == 0:
        return
        
    was_addressed_pct = (feature_stats["candidate_was_addressed_true"] / total) * 100
    missing_recency_pct = (feature_stats["missing_recency"] / total) * 100
    transition_pct = (feature_stats["transition_active"] / total) * 100
    avg_recency = (sum(feature_stats["recencies"]) / len(feature_stats["recencies"])) if feature_stats["recencies"] else 0.0
    
    os.makedirs("results/EXP013", exist_ok=True)
    report_path = "results/EXP013/feature_statistics.md"
    
    with open(report_path, "w") as f:
        f.write("# EXP013A.3 Feature Statistics\n\n")
        f.write(f"Evaluated on {book_name} true speakers.\n\n")
        f.write(f"- **Total quotes evaluated**: {total}\n")
        f.write(f"- **`candidate_was_addressed` true %**: {was_addressed_pct:.2f}%\n")
        f.write(f"- **`speaker_addressee_transition` active %**: {transition_pct:.2f}%\n")
        f.write(f"- **Average `addressee_recency`**: {avg_recency:.2f} turns (excluding missing)\n")
        f.write(f"- **Missing recency %**: {missing_recency_pct:.2f}%\n")
        f.write(f"- **Feature overlap (addressed AND transition)**: {overlap} quotes ({(overlap/total)*100:.2f}%)\n")
        
    print(f"Statistics generated and saved to {report_path}")

if __name__ == "__main__":
    run_feature_statistics()
