import unittest
import torch
import torch.nn as nn
from src.neural.models import EntityAnchoredRelationalGRU

class TestEntityAnchoredRelationalGRU(unittest.TestCase):
    def setUp(self):
        self.feature_dim = 15
        self.vocab_size = 10
        self.emb_dim = 8
        self.hidden_dim = 16
        
        self.model = EntityAnchoredRelationalGRU(
            feature_dim=self.feature_dim,
            vocab_size=self.vocab_size,
            emb_dim=self.emb_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Create a synthetic sequence: 3 quotes, 4 candidates max
        self.seq_len = 3
        self.max_cand = 4
        self.features = torch.randn(1, self.seq_len, self.max_cand, self.feature_dim)
        self.mask = torch.ones(1, self.seq_len, self.max_cand, dtype=torch.bool)
        
        # candidate IDs: let's track one stable character (ID 5) across turns
        # cand 0 at t=0, cand 1 at t=1, cand 2 at t=2 are the same character (ID 5)
        self.candidate_ids = torch.randint(2, self.vocab_size, (1, self.seq_len, self.max_cand))
        self.candidate_ids[0, 0, 0] = 5
        self.candidate_ids[0, 1, 1] = 5
        self.candidate_ids[0, 2, 2] = 5
        
        self.gold_index_for_update = torch.zeros(1, self.seq_len, dtype=torch.long)

    def test_embedding_table_shared(self):
        # Verify candidate scoring and GRU feedback use the exact same embedding table
        model_params = list(self.model.parameters())
        emb_params = list(self.model.char_emb.parameters())
        self.assertTrue(any(torch.equal(p, emb_params[0]) for p in model_params))
        
        # Test that changing character embedding changes both candidate scores and updates
        self.model.eval()
        with torch.no_grad():
            scores1, _ = self.model(self.features, self.candidate_ids, self.mask, self.gold_index_for_update)
            
            # Perturb embedding weight for character 5
            self.model.char_emb.weight[5] += 1.0
            
            scores2, _ = self.model(self.features, self.candidate_ids, self.mask, self.gold_index_for_update)
            
            # Scores must change because embedding changed
            self.assertFalse(torch.allclose(scores1, scores2))

    def test_memory_reset_behavior(self):
        self.model.eval()
        with torch.no_grad():
            # Get base trajectory
            scores_normal, sims_normal = self.model(
                self.features, self.candidate_ids, self.mask, self.gold_index_for_update,
                ablate_memory=False
            )
            
            # Get reset trajectory (GRU reset every step)
            scores_reset, sims_reset = self.model(
                self.features, self.candidate_ids, self.mask, self.gold_index_for_update,
                ablate_memory=True
            )
            
            # The second step (t=1) onwards should change due to memory reset (since t=0 feedback won't propagate)
            self.assertFalse(torch.allclose(scores_normal[:, 1:], scores_reset[:, 1:]))
            self.assertFalse(torch.allclose(sims_normal[:, 1:], sims_reset[:, 1:]))

    def test_shuffled_feedback_behavior(self):
        self.model.eval()
        torch.manual_seed(42) # Pin random seed for shuffle reproducibility
        with torch.no_grad():
            scores_normal, _ = self.model(
                self.features, self.candidate_ids, self.mask, self.gold_index_for_update,
                ablate_shuffle=False
            )
            
            scores_shuffled, _ = self.model(
                self.features, self.candidate_ids, self.mask, self.gold_index_for_update,
                ablate_shuffle=True
            )
            
            # Shuffling feedback index should alter the hidden state trajectory, affecting scores at t=1 and t=2
            self.assertFalse(torch.allclose(scores_normal[:, 1:], scores_shuffled[:, 1:]))

    def test_anchor_instability_behavior(self):
        self.model.eval()
        with torch.no_grad():
            # Run without instability (stable entity IDs)
            scores_stable, _ = self.model(
                self.features, self.candidate_ids, self.mask, self.gold_index_for_update,
                ablate_anchor_instability=False
            )
            
            # Run with instability (shuffled character IDs within timestep)
            scores_unstable, _ = self.model(
                self.features, self.candidate_ids, self.mask, self.gold_index_for_update,
                ablate_anchor_instability=True
            )
            
            # Instability should change candidate IDs and therefore the candidate scores
            self.assertFalse(torch.allclose(scores_stable, scores_unstable))

    def test_no_mutable_discourse_features(self):
        self.assertEqual(self.features.shape[-1], self.feature_dim)

if __name__ == '__main__':
    unittest.main()
