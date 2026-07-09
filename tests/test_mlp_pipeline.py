import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.neural.models import CandidateMLP

def test_mlp_forward_backward():
    print("Running MLP dry-run test...")
    input_dim = 22
    batch_size = 5
    
    # Instantiate model
    model = CandidateMLP(input_dim=input_dim, hidden_dim=64)
    
    # Dummy features and labels
    dummy_input = torch.randn(batch_size, input_dim)
    dummy_labels = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32)
    
    # Forward pass
    logits = model(dummy_input)
    assert logits.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {logits.shape}"
    
    # Loss computation
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(logits, dummy_labels)
    
    # Backward pass
    loss.backward()
    
    print("MLP dry-run test PASSED.")

if __name__ == "__main__":
    test_mlp_forward_backward()
