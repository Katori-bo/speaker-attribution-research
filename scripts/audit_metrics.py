import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

def run_audit():
    df = pd.read_csv("data/raw/pdnc/phase2/candidate_features.csv")
    test_df = df[df['split'] == 'test'].copy()
    train_df = df[df['split'] == 'train'].copy()

    all_features = [c for c in df.columns if c not in ["quote_id", "novel", "candidate", "gold_speaker", "split", "label"] and not c.startswith("symbolic_")]
    
    gbm = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    gbm.fit(train_df[all_features], train_df['label'])
    test_df['score'] = gbm.predict_proba(test_df[all_features])[:, 1]
    
    correct_exp008 = 0
    total_exp008 = 0
    total_quotes = len(test_df['quote_id'].unique())
    
    for _, group in test_df.groupby('quote_id'):
        if group['label'].sum() > 0:
            if group.loc[group['score'].idxmax(), 'label'] == 1:
                correct_exp008 += 1
            total_exp008 += 1
            
    print(f"Total quotes in test: {total_quotes}")
    print(f"Quotes with gold candidate (Oracle): {total_exp008} ({total_exp008/total_quotes:.2%})")
    print(f"EXP008 Accuracy (conditional on Oracle): {correct_exp008 / total_exp008:.4f}")
    print(f"EXP012 Accuracy equivalent (unconditional): {correct_exp008 / total_quotes:.4f}")

if __name__ == "__main__":
    run_audit()
