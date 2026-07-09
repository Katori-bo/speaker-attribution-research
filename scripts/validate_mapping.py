import os
import pandas as pd
import json
from src.coreference.pipeline import SemanticFeatureProvider

def validate_mapping():
    df = pd.read_csv("data/raw/pdnc/phase2/candidate_features.csv")
    provider = SemanticFeatureProvider()
    
    results = []
    
    for novel, novel_df in df.groupby('novel'):
        try:
            # Check if booknlp output exists
            if not os.path.exists(f"data/raw/pdnc/booknlp_out/{novel}/{novel}.book"):
                continue
                
            provider._load_novel(novel)
            
            mapped = 0
            unmapped = 0
            ambiguous = 0
            
            unique_candidates = novel_df['candidate'].unique()
            for candidate in unique_candidates:
                chain_ids = provider.mapper.resolve_string_to_chain_ids(candidate)
                if len(chain_ids) > 0:
                    mapped += 1
                    if len(chain_ids) > 1:
                        ambiguous += 1
                else:
                    unmapped += 1
                    
            total_unique = len(unique_candidates)
            coverage = (mapped / total_unique * 100) if total_unique > 0 else 0
            
            # Count entities and aliases
            booknlp_entities = len(provider.entities) if provider.entities else 0
            alias_count = sum(len(aliases) for aliases in provider.mapper.aliases.values()) if provider.mapper.aliases else 0
            
            results.append({
                "novel": novel,
                "quotes": len(novel_df),
                "candidate_count": total_unique,
                "mapped_candidates": mapped,
                "unmapped_candidates": unmapped,
                "mapping_rate": round(coverage, 2),
                "ambiguous_mappings": ambiguous,
                "booknlp_entities": booknlp_entities,
                "alias_count": alias_count
            })
        except Exception as e:
            print(f"Error processing {novel}: {e}")
            
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # Create results directory if it doesn't exist
        os.makedirs("results/EXP012B", exist_ok=True)
        
        # Save to CSV
        results_df.to_csv("results/EXP012B/mapping_report.csv", index=False)
        
        # Sort and display distribution
        results_df = results_df.sort_values(by="mapping_rate", ascending=False)
        print("\nMapping Coverage Distribution:")
        print("Best novels:")
        print(results_df.head(5)[['novel', 'mapping_rate']].to_string(index=False))
        
        print("\nWorst novels:")
        print(results_df.tail(5)[['novel', 'mapping_rate']].to_string(index=False))
        
        print("\nFull report saved to results/EXP012B/mapping_report.csv")
            
if __name__ == "__main__":
    validate_mapping()
