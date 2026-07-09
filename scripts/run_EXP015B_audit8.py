import os
import pandas as pd
from src.utils.config import get_data_dir

def main():
    exp014_cache = get_data_dir() / "phase2" / "candidate_features_exp014.csv"
    df = pd.read_csv(exp014_cache)
    
    # Load mapping
    quote_types_dict = {}
    all_novels = df['novel'].unique()
    for novel in all_novels:
        q_info_path = get_data_dir() / "data" / novel / "quotation_info.csv"
        if not q_info_path.exists(): continue
        pdnc_quotes = pd.read_csv(q_info_path)
        novel_type_map = {row['quoteID']: str(row['quoteType']) for _, row in pdnc_quotes.iterrows()}
        
        ndf = df[df['novel'] == novel]
        for q_id in ndf['quote_id'].unique():
            pdnc_id = f"Q{q_id.split('_')[-1]}"
            qtype = novel_type_map.get(pdnc_id, 'Implicit')
            if qtype == 'Implicit':
                quote_types_dict[q_id] = 'Implicit'
                
    df['quote_type'] = df['quote_id'].map(quote_types_dict)
    
    implicit_quotes = df[df['quote_type'] == 'Implicit'].groupby('quote_id').first().reset_index()
    if len(implicit_quotes) > 100:
        sampled = implicit_quotes.sample(100, random_state=42)
    else:
        sampled = implicit_quotes
        
    # Analyze patterns by looking at gold speakers of previous quotes
    # To do this accurately, we sort the entire df by novel and quote_id
    # quote_id format: Novel_Idx
    
    patterns = {'Alternation': 0, 'Continuation': 0, 'Multi-party': 0, 'Other': 0}
    
    for _, q in sampled.iterrows():
        novel = q['novel']
        q_id_str = q['quote_id']
        idx = int(q_id_str.split('_')[-1])
        
        # Get gold speakers of previous quotes
        ndf = df[df['novel'] == novel].groupby('quote_id').first()
        
        # We need the index of this quote in the sorted list of quotes for the novel
        def get_idx(s): return int(s.split('_')[-1])
        q_ids_sorted = sorted(ndf.index.tolist(), key=get_idx)
        
        try:
            pos = q_ids_sorted.index(q_id_str)
        except ValueError:
            patterns['Other'] += 1
            continue
            
        if pos < 2:
            patterns['Other'] += 1
            continue
            
        current_speaker = ndf.loc[q_ids_sorted[pos]]['gold_speaker']
        prev_1_speaker = ndf.loc[q_ids_sorted[pos-1]]['gold_speaker']
        prev_2_speaker = ndf.loc[q_ids_sorted[pos-2]]['gold_speaker']
        
        if current_speaker == prev_2_speaker and current_speaker != prev_1_speaker:
            patterns['Alternation'] += 1
        elif current_speaker == prev_1_speaker:
            patterns['Continuation'] += 1
        elif prev_1_speaker != prev_2_speaker and current_speaker not in [prev_1_speaker, prev_2_speaker]:
            patterns['Multi-party'] += 1
        else:
            patterns['Other'] += 1
            
    out_rows = []
    for pat, count in patterns.items():
        out_rows.append({'Pattern': pat, 'Count': count})
        
    pd.DataFrame(out_rows).to_csv("results/EXP015B/implicit_structure_analysis.csv", index=False)
    print("Audit 8 Mechanism Analysis complete.")

if __name__ == "__main__":
    main()
