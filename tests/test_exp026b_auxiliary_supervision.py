import torch
import torch.nn as nn
from src.neural.models import (
    EXP026ABilinearSpeakerGRU,
    EXP026BBilinearSpeakerGRUWithAuxiliary,
    compute_masked_previous_speaker_ce
)

def test_model_dimensions_and_aux_weight():
    model = EXP026BBilinearSpeakerGRUWithAuxiliary(feature_dim=15, hidden_dim=64)
    assert hasattr(model, "aux_bilinear_weight")
    assert model.aux_bilinear_weight.shape == (64, 64)
    assert model.aux_bilinear_weight.numel() == 4096


def test_b_vs_b2_forward_equivalence():
    torch.manual_seed(42)
    # Initialize base bilinear GRU model (B)
    model_b = EXP026ABilinearSpeakerGRU(feature_dim=15, hidden_dim=64)
    
    # Initialize auxiliary model (B2)
    model_b2 = EXP026BBilinearSpeakerGRUWithAuxiliary(feature_dim=15, hidden_dim=64)
    
    # Copy shared weights exactly
    model_b2.load_shared_state_dict(model_b.state_dict())
    
    # Inputs
    feats = torch.randn(1, 5, 4, 15) # [batch, seq_len, max_cand, feature_dim]
    mask = torch.ones(1, 5, 4, dtype=torch.bool)
    gold_index = torch.tensor([[0, 1, 2, 0, 1]])
    
    # Forward pass
    model_b.eval()
    model_b2.eval()
    
    with torch.no_grad():
        scores_b, ints_b = model_b(feats, mask, gold_index_for_update=gold_index)
        scores_b2, ints_b2, aux_scores_b2 = model_b2(feats, mask, gold_index_for_update=gold_index)
        
    assert torch.allclose(scores_b, scores_b2, atol=1e-6)
    assert torch.allclose(ints_b, ints_b2, atol=1e-6)


def test_b_vs_b2_training_equivalence_at_lambda_0():
    torch.manual_seed(42)
    model_b = EXP026ABilinearSpeakerGRU(feature_dim=15, hidden_dim=64)
    model_b2 = EXP026BBilinearSpeakerGRUWithAuxiliary(feature_dim=15, hidden_dim=64)
    
    # Copy shared weights exactly
    model_b2.load_shared_state_dict(model_b.state_dict())
    
    # Define identical optimizers over shared parameters
    # Note: B2's optimizer also includes aux_bilinear_weight, but it receives zero gradient when lambda=0
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1)
    opt_b2 = torch.optim.SGD(model_b2.parameters(), lr=0.1)
    
    # Inputs
    feats = torch.randn(1, 3, 2, 15)
    mask = torch.ones(1, 3, 2, dtype=torch.bool)
    gold_index = torch.tensor([[0, 1, 0]])
    
    # Run training step on model B
    model_b.train()
    opt_b.zero_grad()
    scores_b, _ = model_b(feats, mask, gold_index_for_update=gold_index)
    loss_b = nn.CrossEntropyLoss()(scores_b.squeeze(0), gold_index.squeeze(0))
    loss_b.backward()
    opt_b.step()
    
    # Run training step on model B2 (with lambda = 0)
    model_b2.train()
    opt_b2.zero_grad()
    scores_b2, _, aux_scores_b2 = model_b2(feats, mask, gold_index_for_update=gold_index)
    loss_attr = nn.CrossEntropyLoss()(scores_b2.squeeze(0), gold_index.squeeze(0))
    
    # Calculate mock auxiliary loss
    cids = torch.tensor([[10, 20], [20, 30], [30, 40]])
    gold_spk = torch.tensor([[20, 30, 40]])
    aux_loss = compute_masked_previous_speaker_ce(aux_scores_b2, cids, gold_spk, mask)
    
    total_loss = loss_attr + 0.0 * aux_loss
    total_loss.backward()
    opt_b2.step()
    
    # Ensure W_aux has zero gradient
    assert model_b2.aux_bilinear_weight.grad is None or torch.allclose(model_b2.aux_bilinear_weight.grad, torch.zeros(64, 64))

    # Ensure shared parameter weights match exactly after step
    params_b = dict(model_b.named_parameters())
    params_b2 = dict(model_b2.named_parameters())
    for name, p_b in params_b.items():
        p_b2 = params_b2[name]
        assert torch.allclose(p_b, p_b2, atol=1e-6), f"Mismatch in parameter {name}"


def test_auxiliary_loss_first_quote_masking():
    # Setup step t=0 where gold_speaker_id[t-1] is not defined
    aux_scores = torch.randn(1, 2, 3)
    candidate_ids = torch.tensor([[[1, 2, 3], [1, 2, 3]]])
    gold_speaker_id = torch.tensor([[1, 2]])
    mask = torch.ones(1, 2, 3, dtype=torch.bool)
    
    # We want to verify that no target is computed at step t=0.
    # At t=1, gold_speaker_id[0] is 1. candidate_ids[1, 0] is 1. So it matches.
    # We manually pass only 1 step (which is t=0)
    loss = compute_masked_previous_speaker_ce(aux_scores[:, :1, :], candidate_ids[:, :1, :], gold_speaker_id[:, :1], mask[:, :1, :])
    assert loss.item() == 0.0 # No steps are valid for aux loss when sequence length is 1 (only t=0)


def test_auxiliary_loss_missing_previous_speaker_from_candidates():
    aux_scores = torch.randn(1, 3, 2)
    candidate_ids = torch.tensor([[[10, 20], [30, 40], [50, 60]]])
    gold_speaker_id = torch.tensor([[10, 30, 50]]) # gold at t=0 is 10. gold at t=1 is 30.
    mask = torch.ones(1, 3, 2, dtype=torch.bool)
    
    # At t=1, previous gold speaker is 10. candidate_ids[1] is [30, 40]. 10 is absent.
    # At t=2, previous gold speaker is 30. candidate_ids[2] is [50, 60]. 30 is absent.
    loss = compute_masked_previous_speaker_ce(aux_scores, candidate_ids, gold_speaker_id, mask)
    assert loss.item() == 0.0 # No steps are valid because previous speakers are never in the candidate set


def test_auxiliary_loss_padded_candidate_exclusion():
    aux_scores = torch.zeros(1, 2, 2)
    candidate_ids = torch.tensor([[[10, 20], [10, 20]]])
    gold_speaker_id = torch.tensor([[10, 20]]) # prev spk at t=1 is 10
    
    # At t=1, candidate_ids[1, 0] matches 10, but mask[1, 0] is False (padded candidate)
    mask = torch.tensor([[[True, True], [False, True]]])
    
    loss = compute_masked_previous_speaker_ce(aux_scores, candidate_ids, gold_speaker_id, mask)
    assert loss.item() == 0.0 # Skipped because the only matching candidate was masked out


def test_auxiliary_loss_zero_when_no_valid_targets():
    aux_scores = torch.randn(1, 5, 3)
    candidate_ids = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6]]])
    gold_speaker_id = torch.tensor([[1, 2, 3, 4, 5]])
    mask = torch.ones(1, 5, 3, dtype=torch.bool)
    
    # At t=1, prev gold is 1 (candidate set is [4,5,6] -> absent)
    # At t=2, prev gold is 2 (candidate set is [7,8,9] -> absent)
    # At t=3, prev gold is 3 (candidate set is [1,2,3] -> matches index 2)
    # At t=4, prev gold is 4 (candidate set is [4,5,6] -> matches index 0)
    
    # Let's make sure it's 0.0 when we only look at steps where it doesn't match
    # t=1 and t=2:
    loss = compute_masked_previous_speaker_ce(aux_scores[:, :3, :], candidate_ids[:, :3, :], gold_speaker_id[:, :3], mask[:, :3, :])
    assert loss.item() == 0.0
