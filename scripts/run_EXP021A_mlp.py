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
    
    # 3. Scale features and create datasets & dataloaders
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import TensorDataset, DataLoader
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values.astype(np.float32)
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values.astype(np.float32)
    
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32))
    
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
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
    
    all_probs = []
    with torch.no_grad():
        for (features,) in test_loader:
            features = features.to(device)
            logits = model(features).squeeze(1).cpu()
            probs = torch.sigmoid(logits).numpy()
            if probs.ndim == 0:  # In case batch_size=1
                all_probs.append(float(probs))
            else:
                all_probs.extend(probs.tolist())
                
    pred_df = test_df[['novel', 'quote_id', 'candidate', 'gold_speaker', 'split']].copy()
    pred_df['score'] = all_probs
    
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
