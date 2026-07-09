import torch
import torch.nn as nn

class CandidateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_dim]
        returns logits: [batch_size, 1]
        """
        return self.net(x)

class SpeakerGRU(nn.Module):
    def __init__(self, feature_dim: int, vocab_size: int, emb_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru_cell = nn.GRUCell(emb_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        
        self.candidate_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, candidate_features, candidate_ids, candidate_mask, 
                speaker_ids_for_update=None, ablate_memory=False, ablate_feedback=False):
        if candidate_features.dim() == 4:
            candidate_features = candidate_features.squeeze(0)
            candidate_ids = candidate_ids.squeeze(0)
            candidate_mask = candidate_mask.squeeze(0)
            if speaker_ids_for_update is not None:
                speaker_ids_for_update = speaker_ids_for_update.squeeze(0)
                
        seq_len, max_cand, _ = candidate_features.shape
        device = candidate_features.device
        
        h = torch.zeros(1, self.hidden_dim, device=device)
        all_scores = []
        
        for t in range(seq_len):
            if ablate_memory:
                h = torch.zeros(1, self.hidden_dim, device=device)
                
            feats_t = candidate_features[t]
            cids_t = candidate_ids[t]
            mask_t = candidate_mask[t]
            
            cand_vecs = self.candidate_encoder(feats_t)
            h_expanded = h.expand(max_cand, -1)
            x = torch.cat([cand_vecs, h_expanded], dim=-1)
            
            scores_t = self.scorer(x).squeeze(-1)
            scores_t = scores_t.masked_fill(~mask_t, float('-inf'))
            all_scores.append(scores_t)
            
            if speaker_ids_for_update is not None:
                spk_id = speaker_ids_for_update[t].unsqueeze(0)
            else:
                best_cand_idx = torch.argmax(scores_t)
                spk_id = cids_t[best_cand_idx].unsqueeze(0)
                
            if ablate_feedback:
                spk_emb = torch.zeros(1, self.char_emb.embedding_dim, device=device)
            else:
                spk_emb = self.char_emb(spk_id)
                
            h = self.gru_cell(spk_emb, h)
            
        scores_stacked = torch.stack(all_scores, dim=0)
        return scores_stacked.unsqueeze(0) # [1, seq_len, max_cand]
