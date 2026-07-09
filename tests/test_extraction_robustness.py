import pandas as pd
import numpy as np
from src.utils.config import get_data_dir
from src.features.extractor import FeatureExtractor
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
    csv_path = get_data_dir() / "phase2/candidate_features_exp012.csv"
    df = pd.read_csv(csv_path)
    
    extractor = FeatureExtractor()
    
    for novel, ndf in df.groupby('novel'):
        state = MinimalDiscourseState()
        conv_state = ConversationStateModule(scene_id=f"{novel}_0")
        
        # We need quotation_info.csv to get explicit mentions
        q_info = pd.read_csv(get_data_dir() / "data" / novel / "quotation_info.csv")
        mentions_map = {}
        for _, row in q_info.iterrows():
            m_raw = parse_stringified_list(row.get("mentionEntitiesList", "[]"))
            a_raw = parse_stringified_list(row.get("addressees", "[]"))
            mentions_map[row['quote_id']] = flatten_mentions(m_raw) + flatten_mentions(a_raw)
        
        q_ids = ndf['quote_id'].unique()
        q_ids = sorted(q_ids, key=lambda x: int(x.split('_')[-1]))
        
        for q_id in q_ids:
            qdf = ndf[ndf['quote_id'] == q_id]
            gold = qdf['gold_speaker'].iloc[0]
            
            quote_dict = {
                'quote_text': "dummy", # Can't get this easily from CSV, but lexical feats are static
                'context_text': "dummy",
                'quote_start_byte': int(qdf['quote_start_byte'].iloc[0]),
                'quote_end_byte': int(qdf['quote_end_byte'].iloc[0])
            }
            
            # Wait, FeatureExtractor needs quote_text and context_text for candidate_is_explicit_mention and symbolic rules!
            # If we don't have quote_text in the CSV, we'll have to load the quotes!
            pass

if __name__ == "__main__":
    main()
