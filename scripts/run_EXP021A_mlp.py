import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import yaml
import torch
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.reproducibility import set_seed
from src.utils.device import get_device
from src.evaluation.runner import load_frozen_exp014_dataset
from src.neural.dataset import SpeakerSequenceDataset
from src.neural.dataloader import get_dataloader
from src.neural.models import CandidateMLP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Load config
    with open("configs/EXP021_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(config['seed'])
    
    device = get_device() if config['device'] == 'auto' else config['device']
    logger.info(f"Using device: {device}")
    
    # 2. Load dataset
    logger.info("Loading base dataset...")
    df = load_frozen_exp014_dataset()
    
    # Replicate feature columns extraction
    base_feats = [c for c in df.columns if c not in [
        "quote_id", "novel", "candidate", "gold_speaker", "split", "label",
        "quote_start_byte", "quote_end_byte", "quoteByteSpans",
        "candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency",
        "candidate_is_attributed_speaker"
    ] and not c.startswith("symbolic_")]

    feature_cols = base_feats + [
        "candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency",
        "candidate_is_attributed_speaker"
    ]
    feature_cols = sorted(feature_cols)
    
    # Calculate pos_weight
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    neg_count = len(train_df) - train_df['label'].sum()
    pos_count = train_df['label'].sum()
    pos_weight_val = neg_count / pos_count
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    logger.info(f"Calculated training pos_weight: {pos_weight_val:.4f}")
    
    # 3. Create datasets & dataloaders
    train_dataset = SpeakerSequenceDataset(train_df, feature_cols)
    test_dataset = SpeakerSequenceDataset(test_df, feature_cols)
    
    # MLP doesn't shuffle because sequence dataset parses novels in full, 
    # but we can set shuffle=True to randomize the order of novels in training.
    train_loader = get_dataloader(train_dataset, batch_size=1, shuffle=True)
    test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)
    
    # 4. Initialize model
    input_dim = len(feature_cols)
    model = CandidateMLP(input_dim=input_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Check if this is a CPU test run to limit epochs
    epochs = config['epochs']
    if os.environ.get("CPU_TEST_RUN") == "1":
        epochs = 1
        logger.info("CPU test run: limiting epochs to 1.")
        
    # 5. Training loop
    logger.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            novel_seq = batch[0]
            
            features_list = []
            labels_list = []
            for q in novel_seq.quotes:
                for c in q.candidates:
                    features_list.append(c.features)
                    labels_list.append(float(c.is_gold))
            
            if not features_list:
                continue
                
            features = torch.stack(features_list).to(device)
            labels = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")
        
    # 6. Evaluation
    logger.info("Evaluating on test set...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            novel_seq = batch[0]
            for q in novel_seq.quotes:
                if not q.candidates:
                    continue
                feats = torch.stack([c.features for c in q.candidates]).to(device)
                logits = model(feats).squeeze(1).cpu()
                probs = torch.sigmoid(logits).numpy()
                
                for i, c in enumerate(q.candidates):
                    predictions.append({
                        "novel": novel_seq.novel_id,
                        "quote_id": q.quote_id,
                        "candidate": c.candidate_id,
                        "score": float(probs[i]),
                        "gold_speaker": q.gold_speaker,
                        "split": "test"
                    })
                    
    pred_df = pd.DataFrame(predictions)
    
    # Save predictions
    results_dir = Path("results/EXP021A")
    os.makedirs(results_dir, exist_ok=True)
    pred_df.to_csv(results_dir / "predictions.csv", index=False)
    logger.info(f"Saved test predictions to {results_dir / 'predictions.csv'}")
    
    # Map quote types
    logger.info("Mapping quote types...")
    q_info_dir = Path("data/raw/pdnc/data")
    type_mappings = {}
    for novel in pred_df['novel'].unique():
        q_info_path = q_info_dir / novel / "quotation_info.csv"
        if q_info_path.exists():
            q_info = pd.read_csv(q_info_path)
            for idx, row in q_info.iterrows():
                q_id_raw = row.get("quoteID")
                if pd.isna(q_id_raw) or not q_id_raw:
                    q_id = f"{novel}_{idx}"
                else:
                    quote_num = str(q_id_raw).strip()
                    if quote_num.startswith('Q'):
                        quote_num = quote_num[1:]
                    q_id = f"{novel}_{quote_num}"
                type_mappings[q_id] = row.get("quoteType")
                
    pred_df['quote_type'] = pred_df['quote_id'].map(type_mappings)
    
    # Compute accuracy
    idx_max = pred_df.groupby('quote_id')['score'].idxmax()
    predictions_best = pred_df.loc[idx_max]
    
    overall_accuracy = (predictions_best['candidate'] == predictions_best['gold_speaker']).mean()
    
    implicit_preds = predictions_best[predictions_best['quote_type'] != 'Explicit']
    implicit_accuracy = (implicit_preds['candidate'] == implicit_preds['gold_speaker']).mean()
    
    metrics = {
        "overall_accuracy": float(overall_accuracy),
        "implicit_accuracy": float(implicit_accuracy),
        "num_test_quotes": int(len(predictions_best)),
        "num_test_implicit_quotes": int(len(implicit_preds))
    }
    
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Evaluation Complete:")
    logger.info(f"- Overall Accuracy: {overall_accuracy*100:.2f}%")
    logger.info(f"- Implicit Accuracy: {implicit_accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
