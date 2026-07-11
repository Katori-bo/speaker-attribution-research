import os
import json
import torch
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from src.utils.device import get_device
from src.evaluation.runner import load_frozen_exp014_dataset
from src.neural.sequence_dataset import build_character_vocab, TensorSequenceDataset
from src.neural.models import NoMemoryEntityScorer
from scripts.run_EXP021A_2_mlp_ce import compute_metrics
from scripts.run_EXP023_entity_binding import evaluate_scorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    os.makedirs("results/EXP023", exist_ok=True)
    device = get_device()
    
    df = load_frozen_exp014_dataset()
    
    # 1. EXP023 features
    exp023_base_feats = [c for c in df.columns if c not in [
        "quote_id", "novel", "candidate", "gold_speaker", "split", "label",
        "candidate_id", "novel_candidate", "novel_gold"
    ]]
    mutable_discourse_features = [
        'candidate_is_last_speaker',
        'candidate_is_previous_speaker',
        'candidate_in_participant_stack',
        'candidate_stack_depth',
        'conversation_speaker_change',
        'conv_active_id',
        'conv_interruption_distance'
    ]
    exp023_state_free_cols = sorted([c for c in exp023_base_feats if c not in mutable_discourse_features])
    
    # 2. EXP021A.2 features
    exp021a2_base_feats = [c for c in df.columns if c not in [
        "quote_id", "novel", "candidate", "gold_speaker", "split", "label",
        "quote_start_byte", "quote_end_byte", "quoteByteSpans",
        "candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency",
        "candidate_is_attributed_speaker"
    ] and not c.startswith("symbolic_")]
    exp021a2_feature_cols = sorted(exp021a2_base_feats + [
        "candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency",
        "candidate_is_attributed_speaker"
    ])
    exp021a2_state_free_cols = sorted([c for c in exp021a2_feature_cols if c not in mutable_discourse_features])
    
    # Assert no mutable discourse features are in either
    for f in mutable_discourse_features:
        assert f not in exp023_state_free_cols, f"{f} leaked into EXP023 features!"
        assert f not in exp021a2_state_free_cols, f"{f} leaked into EXP021A.2 features!"
        
    diff_23_has_not_21 = set(exp023_state_free_cols) - set(exp021a2_state_free_cols)
    diff_21_has_not_23 = set(exp021a2_state_free_cols) - set(exp023_state_free_cols)
    
    # Hash of dataframe
    df_hash = pd.util.hash_pandas_object(df).sum()
    
    # Rerun no_anchor with exp021a2_state_free_cols
    vocab = build_character_vocab(df, vocab_path="results/EXP023/character_vocab_audit.json")
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    scaler = StandardScaler()
    train_df[exp021a2_state_free_cols] = scaler.fit_transform(train_df[exp021a2_state_free_cols])
    test_df[exp021a2_state_free_cols] = scaler.transform(test_df[exp021a2_state_free_cols])
    
    train_seq = TensorSequenceDataset(train_df, exp021a2_state_free_cols, feature_mode='all', vocab=vocab)
    test_seq = TensorSequenceDataset(test_df, exp021a2_state_free_cols, feature_mode='all', vocab=vocab)
    
    def collate_fn(batch): return batch[0]
    train_loader = DataLoader(train_seq, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_seq, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    input_dim = train_seq[0].candidate_features.shape[-1]
    
    model = NoMemoryEntityScorer(
        feature_dim=input_dim, 
        vocab_size=len(vocab),
        emb_dim=32,
        hidden_dim=128, 
        anchor_mode='no_anchor'
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 10
    if os.environ.get("CPU_TEST_RUN") == "1":
        epochs = 1
        
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            features = batch.candidate_features.unsqueeze(0).to(device)
            cids = batch.candidate_ids.unsqueeze(0).to(device)
            mask = batch.candidate_mask.unsqueeze(0).to(device)
            gold_index = batch.gold_index.to(device)
            
            optimizer.zero_grad()
            scores, _ = model(features, cids, mask)
            scores = scores.squeeze(0)
            loss = criterion(scores, gold_index)
            loss.backward()
            optimizer.step()
            
    # Load quote types
    q_info_dir = Path("data/raw/pdnc/data")
    type_mappings = {}
    for novel in test_df['novel'].unique():
        q_info_path = q_info_dir / novel / "quotation_info.csv"
        if q_info_path.exists():
            q_info = pd.read_csv(q_info_path)
            for idx, row in q_info.iterrows():
                q_id_raw = row.get("quoteID")
                if pd.isna(q_id_raw) or not q_id_raw: q_id = f"{novel}_{idx}"
                else:
                    quote_num = str(q_id_raw).strip()
                    if quote_num.startswith('Q'): quote_num = quote_num[1:]
                    q_id = f"{novel}_{quote_num}"
                type_mappings[q_id] = row.get("quoteType")
                
    preds = evaluate_scorer(model, test_loader, device, type_mappings)
    mets = compute_metrics(preds)
    
    audit_results = {
        "exp023_features_used": exp023_state_free_cols,
        "exp021a2_features_used": exp021a2_state_free_cols,
        "features_in_23_not_21": list(diff_23_has_not_21),
        "features_in_21_not_23": list(diff_21_has_not_23),
        "dataset_hash": str(df_hash),
        "train_quote_count": len(train_df['quote_id'].unique()),
        "test_quote_count": len(test_df['quote_id'].unique()),
        "train_candidate_count": len(train_df),
        "test_candidate_count": len(test_df),
        "no_anchor_with_21a2_features_acc": float(mets['Accuracy']),
        "no_anchor_with_21a2_features_implicit_acc": float(mets['Implicit_Accuracy'])
    }
    
    with open("results/EXP023/feature_equivalence_audit.json", "w") as f:
        json.dump(audit_results, f, indent=4)
        
    print(json.dumps(audit_results, indent=4))

if __name__ == "__main__":
    main()
