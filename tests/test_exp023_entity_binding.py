import pytest
import torch
import hashlib
from src.neural.models import NoMemoryEntityScorer

@pytest.fixture
def dummy_inputs():
    batch_size = 1
    seq_len = 3
    max_cand = 4
    feature_dim = 5
    vocab_size = 10
    
    features = torch.randn(seq_len, max_cand, feature_dim)
    cids = torch.randint(1, vocab_size, (seq_len, max_cand))
    mask = torch.ones(seq_len, max_cand, dtype=torch.bool)
    mask[:, -1] = False # Make last candidate invalid for testing
    quote_ids = [f"q_{i}" for i in range(seq_len)]
    
    return features, cids, mask, quote_ids, feature_dim, vocab_size

def test_frozen_persistent(dummy_inputs):
    features, cids, mask, quote_ids, feature_dim, vocab_size = dummy_inputs
    model = NoMemoryEntityScorer(feature_dim, vocab_size, anchor_mode='frozen_persistent')
    
    assert not model.char_emb.weight.requires_grad
    
    # Same vector for same entity across timesteps?
    # Make sure we have the same ID at different timesteps
    cids[0, 0] = 5
    cids[1, 0] = 5
    
    scores, anchors = model(features, cids, mask, quote_ids)
    assert torch.allclose(anchors[0, 0, 0], anchors[0, 1, 0])
    
def test_trainable_persistent(dummy_inputs):
    features, cids, mask, quote_ids, feature_dim, vocab_size = dummy_inputs
    model = NoMemoryEntityScorer(feature_dim, vocab_size, anchor_mode='trainable_persistent')
    
    assert model.char_emb.weight.requires_grad
    
    cids[0, 0] = 5
    cids[1, 0] = 5
    
    scores, anchors = model(features, cids, mask, quote_ids)
    assert torch.allclose(anchors[0, 0, 0], anchors[0, 1, 0])

def test_ephemeral(dummy_inputs):
    features, cids, mask, quote_ids, feature_dim, vocab_size = dummy_inputs
    model = NoMemoryEntityScorer(feature_dim, vocab_size, anchor_mode='ephemeral')
    
    cids[0, 0] = 5
    cids[1, 0] = 5
    
    # Same entity, different quote
    scores1, anchors1 = model(features, cids, mask, quote_ids)
    
    # Vector should be different for the same entity in different quotes
    assert not torch.allclose(anchors1[0, 0, 0], anchors1[0, 1, 0])
    
    # Vector should be the same on rerun with same quote
    scores2, anchors2 = model(features, cids, mask, quote_ids)
    assert torch.allclose(anchors1[0, 0, 0], anchors2[0, 0, 0])

def test_shuffled_persistent(dummy_inputs):
    features, cids, mask, quote_ids, feature_dim, vocab_size = dummy_inputs
    model = NoMemoryEntityScorer(feature_dim, vocab_size, anchor_mode='shuffled_persistent')
    
    cids[0, 0] = 5
    cids[1, 0] = 5
    
    scores, anchors = model(features, cids, mask, quote_ids)
    assert torch.allclose(anchors[0, 0, 0], anchors[0, 1, 0])
    
    # Verify mapping is different from identity (most of the time)
    mapped_id = model.shuffled_mapping[5].item()
    assert mapped_id != 0 # padding untouched
    
def test_constant(dummy_inputs):
    features, cids, mask, quote_ids, feature_dim, vocab_size = dummy_inputs
    model = NoMemoryEntityScorer(feature_dim, vocab_size, anchor_mode='constant')
    
    assert not model.constant_vector.requires_grad
    
    scores, anchors = model(features, cids, mask, quote_ids)
    
    # All valid anchors should be exactly the same
    valid_anchor_0 = anchors[0, 0, 0]
    valid_anchor_1 = anchors[0, 0, 1]
    
    assert torch.allclose(valid_anchor_0, valid_anchor_1)
    assert torch.allclose(valid_anchor_0, anchors[0, 1, 0])
    
def test_position(dummy_inputs):
    features, cids, mask, quote_ids, feature_dim, vocab_size = dummy_inputs
    model = NoMemoryEntityScorer(feature_dim, vocab_size, anchor_mode='position')
    
    scores, anchors = model(features, cids, mask, quote_ids)
    
    # Anchors should depend on position, not entity id
    cids[0, 0] = 3
    cids[0, 1] = 5
    
    assert not torch.allclose(anchors[0, 0, 0], anchors[0, 0, 1])
    # Position 0 in quote 0 should match position 0 in quote 1
    assert torch.allclose(anchors[0, 0, 0], anchors[0, 1, 0])
    
def test_deterministic_hash(dummy_inputs):
    features, cids, mask, quote_ids, feature_dim, vocab_size = dummy_inputs
    
    hash_embs = torch.randn(vocab_size, 32)
    for i in range(1, vocab_size):
        seed_str = f"hash_char_{i}"
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
        gen = torch.Generator(device='cpu').manual_seed(seed)
        hash_embs[i] = torch.randn(32, generator=gen)
        
    model = NoMemoryEntityScorer(feature_dim, vocab_size, anchor_mode='deterministic_hash', pretrained_emb=hash_embs)
    
    assert not model.char_emb.weight.requires_grad
    
    cids[0, 0] = 2
    cids[1, 0] = 2
    cids[0, 1] = 3
    
    scores, anchors = model(features, cids, mask, quote_ids)
    assert torch.allclose(anchors[0, 0, 0], anchors[0, 1, 0])
    assert not torch.allclose(anchors[0, 0, 0], anchors[0, 0, 1])
    
def test_unstable(dummy_inputs):
    features, cids, mask, quote_ids, feature_dim, vocab_size = dummy_inputs
    model = NoMemoryEntityScorer(feature_dim, vocab_size, anchor_mode='unstable')
    
    cids[0, 0] = 5
    cids[1, 0] = 5
    
    scores1, anchors1 = model(features, cids, mask, quote_ids)
    
    # Same entity, different quote -> different vector
    assert not torch.allclose(anchors1[0, 0, 0], anchors1[0, 1, 0])
    
    # Same rerun -> same vector
    scores2, anchors2 = model(features, cids, mask, quote_ids)
    assert torch.allclose(anchors1[0, 0, 0], anchors2[0, 0, 0])
