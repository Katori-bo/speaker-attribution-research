import pandas as pd
from src.utils.config import get_data_dir
from src.evaluation.dynamic_features import extract_dynamic_features
from src.discourse.discourse_state import MinimalDiscourseState
from src.discourse.conversation_state import ConversationStateModule
import ast

def parse_stringified_list(val):
    try: return ast.literal_eval(val)
    except: return []

def flatten_mentions(mentions):
    flat = []
    if isinstance(mentions, list):
        for item in mentions:
            if isinstance(item, list): flat.extend(flatten_mentions(item))
            else: flat.append(item)
    return flat

def main():
    exp012_cache_file = get_data_dir() / "phase2" / "candidate_features_exp012.csv"
    df = pd.read_csv(exp012_cache_file)
    test_df = df[df['split'] == 'test'].copy()
    
    mismatches = {}
    
    for novel, novel_df in test_df.groupby('novel'):
        state = MinimalDiscourseState()
        conv_state = ConversationStateModule(scene_id=f"{novel}_0")
        
        q_info_path = get_data_dir() / "data" / novel / "quotation_info.csv"
        q_info = pd.read_csv(q_info_path)
        q_info_map = {row['quote_id'] if 'quote_id' in row else f"{novel}_{i}": row for i, row in q_info.iterrows()}
        
        unique_quotes = sorted(novel_df['quote_id'].unique(), key=lambda x: int(x.split('_')[-1]))
        gold_previous_speakers = []
        
        for idx_in_novel, q_row in q_info.iterrows():
            q_id = q_row.get("quote_id")
            if not q_id: q_id = f"{novel}_{idx_in_novel}"
            
            gold_speaker = str(q_row.get("speaker", "Unknown")).strip()
            
            mentions_raw = parse_stringified_list(q_row.get("mentionEntitiesList", "[]"))
            addressees_raw = parse_stringified_list(q_row.get("addressees", "[]"))
            explicit_mentions = flatten_mentions(mentions_raw) + flatten_mentions(addressees_raw)
            
            q_df = novel_df[novel_df['quote_id'] == q_id].copy()
            if not q_df.empty:
                candidates_set = set(q_df['candidate'].unique())
            else:
                candidates_set = set()
            
            if not q_df.empty:
                df_dp = q_df.iloc[0]['discourse_dialogue_position']
                if df_dp == 1.0:
                    conv_state.reset(novel)
            
            gold_prev = gold_previous_speakers[-1] if gold_previous_speakers else None
                    
            # The original script does state.update before extracting features!
            state.update(gold_prev, explicit_mentions, candidates_set)
            
            quote_spans_raw = parse_stringified_list(q_row.get("quoteByteSpans", "[]"))
            q_start = -1
            q_end = -1
            if quote_spans_raw and len(quote_spans_raw) > 0:
                try:
                    q_start = int(quote_spans_raw[0][0])
                    q_end = int(quote_spans_raw[-1][1])
                except:
                    pass
            
            if not q_df.empty:
                df_q_start = int(q_df['quote_start_byte'].iloc[0])
                for idx, row in q_df.iterrows():
                    candidate = row['candidate']
                    dyn_feats = extract_dynamic_features(candidate, state, conv_state, df_q_start)
                    
                    # Check for mismatches
                    for k, v in dyn_feats.items():
                        if k in row:
                            orig = row[k]
                            if orig != v and not (pd.isna(orig) and pd.isna(v)):
                                if k not in mismatches: mismatches[k] = 0
                                mismatches[k] += 1
                                if mismatches[k] < 5:
                                    print(f"Mismatch in {k} for {q_id} {candidate}: Orig={orig} New={v}")
            
            conv_state.update({"quote_start_byte": q_start, "quote_end_byte": q_end}, gold_speaker)
            
            if gold_speaker != "Unknown":
                gold_previous_speakers.append(gold_speaker)
                
    print(f"Mismatches: {mismatches}")

if __name__ == "__main__":
    main()
