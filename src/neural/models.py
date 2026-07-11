import torch
import torch.nn as nn
import hashlib

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
            nn.Linear(feature_dim + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, candidate_features, candidate_ids, candidate_mask, 
                speaker_ids_for_update=None, ablate_memory=False, ablate_feedback=False, ablate_shuffle=False):
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
            
            cand_embs = self.char_emb(cids_t)
            cand_inputs = torch.cat([feats_t, cand_embs], dim=-1)
            cand_vecs = self.candidate_encoder(cand_inputs)
            
            h_expanded = h.expand(max_cand, -1)
            x = torch.cat([cand_vecs, h_expanded], dim=-1)
            
            scores_t = self.scorer(x).squeeze(-1)
            scores_t = scores_t.masked_fill(~mask_t, float('-inf'))
            all_scores.append(scores_t)
            
            if ablate_shuffle:
                num_valid = mask_t.sum().item()
                if num_valid > 0:
                    rand_idx = torch.randint(0, num_valid, (1,)).item()
                    spk_id = cids_t[rand_idx].unsqueeze(0)
                else:
                    spk_id = cids_t[0].unsqueeze(0)
            elif speaker_ids_for_update is not None:
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
        return scores_stacked.unsqueeze(0)

class EntityAnchoredRelationalGRU(nn.Module):
    def __init__(self, feature_dim: int, vocab_size: int, emb_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        
        self.char_emb = nn.Embedding(vocab_size, emb_dim)
        
        self.candidate_encoder = nn.Sequential(
            nn.Linear(feature_dim + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, candidate_features, candidate_ids, candidate_mask, 
                gold_index_for_update=None, ablate_memory=False, ablate_shuffle=False, 
                ablate_similarity=False, ablate_anchor_instability=False):
        if candidate_features.dim() == 4:
            candidate_features = candidate_features.squeeze(0)
            candidate_ids = candidate_ids.squeeze(0)
            candidate_mask = candidate_mask.squeeze(0)
            if gold_index_for_update is not None:
                gold_index_for_update = gold_index_for_update.squeeze(0)
                
        seq_len, max_cand, _ = candidate_features.shape
        device = candidate_features.device
        
        h = torch.zeros(1, self.hidden_dim, device=device)
        all_scores = []
        all_sims = []
        
        cids_working = candidate_ids.clone()
        if ablate_anchor_instability:
            for t in range(seq_len):
                mask_t = candidate_mask[t]
                num_valid = mask_t.sum().item()
                if num_valid > 1:
                    valid_ids = cids_working[t, mask_t]
                    shuffled_valid_ids = valid_ids[torch.randperm(num_valid)]
                    cids_working[t, mask_t] = shuffled_valid_ids
        
        for t in range(seq_len):
            if ablate_memory:
                h = torch.zeros(1, self.hidden_dim, device=device)
                
            feats_t = candidate_features[t]
            mask_t = candidate_mask[t]
            cids_t = cids_working[t]
            
            cand_embs = self.char_emb(cids_t)
            cand_inputs = torch.cat([feats_t, cand_embs], dim=-1)
            cand_vecs = self.candidate_encoder(cand_inputs)
            
            h_expanded = h.expand(max_cand, -1)
            
            sim = torch.nn.functional.cosine_similarity(cand_vecs, h_expanded, dim=-1).unsqueeze(-1)
            if ablate_similarity:
                sim = torch.zeros_like(sim)
                
            x = torch.cat([cand_vecs, h_expanded, sim], dim=-1)
            scores_t = self.scorer(x).squeeze(-1)
            scores_t = scores_t.masked_fill(~mask_t, float('-inf'))
            all_scores.append(scores_t)
            all_sims.append(sim)
            
            if gold_index_for_update is not None:
                spk_idx = gold_index_for_update[t].item()
            else:
                spk_idx = torch.argmax(scores_t).item()
                
            if ablate_shuffle:
                num_valid = mask_t.sum().item()
                if num_valid > 0:
                    spk_idx = torch.randint(0, num_valid, (1,)).item()
                else:
                    spk_idx = 0
                    
            spk_vec = cand_vecs[spk_idx].unsqueeze(0)
            h = self.gru_cell(spk_vec, h)
            
        scores_stacked = torch.stack(all_scores, dim=0).unsqueeze(0)
        sims_stacked = torch.stack(all_sims, dim=0).unsqueeze(0)
        return scores_stacked, sims_stacked # [1, seq_len, max_cand]

class RelationalSpeakerGRU(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.candidate_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, candidate_features, candidate_mask, 
                gold_index_for_update=None, ablate_memory=False, ablate_shuffle=False, ablate_similarity=False):
        if candidate_features.dim() == 4:
            candidate_features = candidate_features.squeeze(0)
            candidate_mask = candidate_mask.squeeze(0)
            if gold_index_for_update is not None:
                gold_index_for_update = gold_index_for_update.squeeze(0)
                
        seq_len, max_cand, _ = candidate_features.shape
        device = candidate_features.device
        
        h = torch.zeros(1, self.hidden_dim, device=device)
        all_scores = []
        all_sims = []
        
        for t in range(seq_len):
            if ablate_memory:
                h = torch.zeros(1, self.hidden_dim, device=device)
                
            feats_t = candidate_features[t]
            mask_t = candidate_mask[t]
            
            cand_vecs = self.candidate_encoder(feats_t)
            h_expanded = h.expand(max_cand, -1)
            
            sim = torch.nn.functional.cosine_similarity(cand_vecs, h_expanded, dim=-1).unsqueeze(-1)
            if ablate_similarity:
                sim = torch.zeros_like(sim)
            all_sims.append(sim)
                
            x = torch.cat([cand_vecs, h_expanded, sim], dim=-1)
            scores_t = self.scorer(x).squeeze(-1)
            scores_t = scores_t.masked_fill(~mask_t, float('-inf'))
            all_scores.append(scores_t)
            
            if gold_index_for_update is not None:
                spk_idx = gold_index_for_update[t].item()
            else:
                spk_idx = torch.argmax(scores_t).item()
                
            if ablate_shuffle:
                num_valid = mask_t.sum().item()
                if num_valid > 0:
                    spk_idx = torch.randint(0, num_valid, (1,)).item()
                else:
                    spk_idx = 0
                    
            spk_vec = cand_vecs[spk_idx].unsqueeze(0)
            h = self.gru_cell(spk_vec, h)
            
        scores_stacked = torch.stack(all_scores, dim=0)
        sims_stacked = torch.stack(all_sims, dim=0)
        return scores_stacked.unsqueeze(0), sims_stacked.unsqueeze(0)

class EXP026ACandidateOnlyScorer(nn.Module):
    """
    EXP026A Variant A.

    Scores candidates through the same candidate encoder and candidate-only branch
    used by EXP026ABilinearSpeakerGRU. There is no recurrent state in this model.
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_names = None

        self.candidate_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.candidate_score_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def encode_candidates(self, candidate_features: torch.Tensor) -> torch.Tensor:
        return self.candidate_encoder(candidate_features)

    def candidate_score(self, candidate_repr: torch.Tensor) -> torch.Tensor:
        return self.candidate_score_branch(candidate_repr).squeeze(-1)

    def score(self, candidate_repr: torch.Tensor) -> torch.Tensor:
        return self.candidate_score(candidate_repr)

    def forward(self, candidate_features, candidate_mask):
        if candidate_features.dim() == 4:
            candidate_features = candidate_features.squeeze(0)
            candidate_mask = candidate_mask.squeeze(0)

        seq_len, _, _ = candidate_features.shape
        all_scores = []

        for t in range(seq_len):
            cand_vecs = self.encode_candidates(candidate_features[t])
            scores_t = self.score(cand_vecs)
            scores_t = scores_t.masked_fill(~candidate_mask[t], float('-inf'))
            all_scores.append(scores_t)

        return torch.stack(all_scores, dim=0).unsqueeze(0), None

class EXP026AParameterMatchedNoMemoryScorer(nn.Module):
    """
    Secondary EXP026A capacity control.

    This model keeps the candidate encoder state-free and adds no recurrent path.
    Its scorer width can be increased to approximate the trainable parameter count
    of the bilinear GRU when the preregistered 10% threshold requires a capacity
    control.
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 64, scorer_hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scorer_hidden_dim = scorer_hidden_dim
        self.feature_names = None

        self.candidate_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.candidate_score_branch = nn.Sequential(
            nn.Linear(hidden_dim, scorer_hidden_dim),
            nn.ReLU(),
            nn.Linear(scorer_hidden_dim, scorer_hidden_dim),
            nn.ReLU(),
            nn.Linear(scorer_hidden_dim, 1)
        )

    def encode_candidates(self, candidate_features: torch.Tensor) -> torch.Tensor:
        return self.candidate_encoder(candidate_features)

    def candidate_score(self, candidate_repr: torch.Tensor) -> torch.Tensor:
        return self.candidate_score_branch(candidate_repr).squeeze(-1)

    def score(self, candidate_repr: torch.Tensor) -> torch.Tensor:
        return self.candidate_score(candidate_repr)

    def forward(self, candidate_features, candidate_mask):
        if candidate_features.dim() == 4:
            candidate_features = candidate_features.squeeze(0)
            candidate_mask = candidate_mask.squeeze(0)

        seq_len, _, _ = candidate_features.shape
        all_scores = []

        for t in range(seq_len):
            cand_vecs = self.encode_candidates(candidate_features[t])
            scores_t = self.score(cand_vecs)
            scores_t = scores_t.masked_fill(~candidate_mask[t], float('-inf'))
            all_scores.append(scores_t)

        return torch.stack(all_scores, dim=0).unsqueeze(0), None

class EXP026ABilinearSpeakerGRU(nn.Module):
    """
    EXP026A Variant C.

    The scorer is exactly:
        s(c, h) = f(c) + c^T W h

    f(c) is the full candidate-only branch used by EXP026ACandidateOnlyScorer.
    The interaction branch has no bias; only W is trainable. Therefore
    s(c, 0) = f(c) by construction.
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_names = None

        self.candidate_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        self.candidate_score_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.bilinear_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.bilinear_weight)

    def encode_candidates(self, candidate_features: torch.Tensor) -> torch.Tensor:
        return self.candidate_encoder(candidate_features)

    def candidate_score(self, candidate_repr: torch.Tensor) -> torch.Tensor:
        return self.candidate_score_branch(candidate_repr).squeeze(-1)

    def interaction(self, candidate_repr: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        if hidden.shape[0] == 1 and candidate_repr.shape[0] != 1:
            hidden = hidden.expand(candidate_repr.shape[0], -1)
        wh = torch.matmul(hidden, self.bilinear_weight.t())
        return (candidate_repr * wh).sum(dim=-1)

    def score(self, candidate_repr: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return self.candidate_score(candidate_repr) + self.interaction(candidate_repr, hidden)

    def forward(self, candidate_features, candidate_mask,
                gold_index_for_update=None, ablate_memory=False, ablate_shuffle=False,
                return_interactions=False):
        if candidate_features.dim() == 4:
            candidate_features = candidate_features.squeeze(0)
            candidate_mask = candidate_mask.squeeze(0)
            if gold_index_for_update is not None:
                gold_index_for_update = gold_index_for_update.squeeze(0)

        seq_len, max_cand, _ = candidate_features.shape
        device = candidate_features.device

        h = torch.zeros(1, self.hidden_dim, device=device)
        all_scores = []
        all_interactions = []

        for t in range(seq_len):
            if ablate_memory:
                h = torch.zeros(1, self.hidden_dim, device=device)

            feats_t = candidate_features[t]
            mask_t = candidate_mask[t]

            cand_vecs = self.encode_candidates(feats_t)
            h_expanded = h.expand(max_cand, -1)
            interactions_t = self.interaction(cand_vecs, h_expanded)
            scores_t = self.score(cand_vecs, h_expanded)
            scores_t = scores_t.masked_fill(~mask_t, float('-inf'))

            all_scores.append(scores_t)
            all_interactions.append(interactions_t.masked_fill(~mask_t, 0.0))

            if gold_index_for_update is not None:
                spk_idx = gold_index_for_update[t].item()
            else:
                spk_idx = torch.argmax(scores_t).item()

            if ablate_shuffle:
                valid_indices = torch.nonzero(mask_t, as_tuple=False).flatten()
                if len(valid_indices) > 0:
                    spk_idx = valid_indices[torch.randint(0, len(valid_indices), (1,), device=device)].item()
                else:
                    spk_idx = 0

            spk_vec = cand_vecs[spk_idx].unsqueeze(0)
            h = self.gru_cell(spk_vec, h)

        scores_stacked = torch.stack(all_scores, dim=0).unsqueeze(0)
        interactions_stacked = torch.stack(all_interactions, dim=0).unsqueeze(0)

        if return_interactions:
            return scores_stacked, interactions_stacked
        return scores_stacked, interactions_stacked

class NoMemoryEntityScorer(nn.Module):
    def __init__(self, feature_dim: int, vocab_size: int, emb_dim: int = 32, hidden_dim: int = 64, 
                 anchor_mode: str = 'trainable_persistent', pretrained_emb=None):
        super().__init__()
        self.anchor_mode = anchor_mode
        self.emb_dim = emb_dim
        
        self.char_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        if pretrained_emb is not None:
            self.char_emb.weight.data.copy_(pretrained_emb)
            
        if self.anchor_mode != 'trainable_persistent':
            self.char_emb.weight.requires_grad = False
            
        if self.anchor_mode == 'constant':
            self.constant_vector = nn.Parameter(torch.randn(1, emb_dim), requires_grad=False)
            
        self.pos_emb = nn.Embedding(200, emb_dim)
        
        if self.anchor_mode == 'shuffled_persistent':
            gen = torch.Generator().manual_seed(42)
            perm = torch.randperm(vocab_size - 1, generator=gen) + 1
            mapping = torch.arange(vocab_size)
            mapping[1:] = perm
            self.register_buffer('shuffled_mapping', mapping)
            
        self.ephemeral_base_seed = 42
        self.ephemeral_cache = {}
        self.unstable_cache = {}
        
        scorer_input_dim = feature_dim if self.anchor_mode == 'no_anchor' else feature_dim + emb_dim
            
        self.scorer = nn.Sequential(
            nn.Linear(scorer_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, candidate_features, candidate_ids, candidate_mask, quote_ids=None):
        if candidate_features.dim() == 4:
            candidate_features = candidate_features.squeeze(0)
            candidate_ids = candidate_ids.squeeze(0)
            candidate_mask = candidate_mask.squeeze(0)
            
        seq_len, max_cand, _ = candidate_features.shape
        device = candidate_features.device
        
        all_scores = []
        all_anchors = []
        
        for t in range(seq_len):
            feats_t = candidate_features[t]
            cids_t = candidate_ids[t]
            mask_t = candidate_mask[t]
            
            if self.anchor_mode == 'no_anchor':
                anchor_t = None
            elif self.anchor_mode == 'constant':
                anchor_t = self.constant_vector.expand(max_cand, -1)
            elif self.anchor_mode == 'position':
                anchor_t = self.pos_emb(torch.arange(max_cand, device=device))
            elif self.anchor_mode == 'ephemeral':
                q_id = quote_ids[t] if quote_ids is not None else str(t)
                anchor_t = torch.zeros(max_cand, self.emb_dim, device=device)
                for c in range(max_cand):
                    if mask_t[c]:
                        eid = cids_t[c].item()
                        key = (eid, q_id)
                        if key not in self.ephemeral_cache:
                            seed_str = f"{eid}_{q_id}_{self.ephemeral_base_seed}"
                            seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
                            gen = torch.Generator(device='cpu').manual_seed(seed)
                            self.ephemeral_cache[key] = torch.randn(self.emb_dim, generator=gen)
                        anchor_t[c] = self.ephemeral_cache[key].to(device)
            elif self.anchor_mode == 'unstable':
                q_id = quote_ids[t] if quote_ids is not None else str(t)
                anchor_t = torch.zeros(max_cand, self.emb_dim, device=device)
                for c in range(max_cand):
                    if mask_t[c]:
                        eid = cids_t[c].item()
                        key = (eid, q_id)
                        if key not in self.unstable_cache:
                            seed_str = f"unstable_{eid}_{q_id}"
                            seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
                            gen = torch.Generator(device='cpu').manual_seed(seed)
                            fake_id = torch.randint(1, self.char_emb.num_embeddings, (1,), generator=gen).item()
                            self.unstable_cache[key] = self.char_emb(torch.tensor(fake_id, device=device)).detach()
                        anchor_t[c] = self.unstable_cache[key].to(device)
            elif self.anchor_mode == 'shuffled_persistent':
                mapped_ids = self.shuffled_mapping[cids_t]
                anchor_t = self.char_emb(mapped_ids)
            else:
                # trainable_persistent, frozen_persistent, deterministic_hash
                anchor_t = self.char_emb(cids_t)
                
            if anchor_t is not None:
                cand_inputs = torch.cat([feats_t, anchor_t], dim=-1)
            else:
                cand_inputs = feats_t
                
            scores_t = self.scorer(cand_inputs).squeeze(-1)
            scores_t = scores_t.masked_fill(~mask_t, float('-inf'))
            all_scores.append(scores_t)
            all_anchors.append(anchor_t if anchor_t is not None else torch.zeros(max_cand, self.emb_dim, device=device))
            
        scores_stacked = torch.stack(all_scores, dim=0)
        anchors_stacked = torch.stack(all_anchors, dim=0) 
        
        return scores_stacked.unsqueeze(0), anchors_stacked.unsqueeze(0)


class EXP026BBilinearSpeakerGRUWithAuxiliary(nn.Module):
    """
    EXP026B Variant C & B2.
    Scorer is s(c, h) = f(c) + c^T W h.
    Also computes auxiliary scores for previous-speaker prediction: aux(c, h) = c^T W_aux h.
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_names = None

        self.candidate_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        self.candidate_score_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.bilinear_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.bilinear_weight)

        self.aux_bilinear_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.aux_bilinear_weight)

    def load_shared_state_dict(self, state_dict):
        shared_state_dict = {k: v for k, v in state_dict.items() if "aux_bilinear" not in k}
        self.load_state_dict(shared_state_dict, strict=False)

    def encode_candidates(self, candidate_features: torch.Tensor) -> torch.Tensor:
        return self.candidate_encoder(candidate_features)

    def candidate_score(self, candidate_repr: torch.Tensor) -> torch.Tensor:
        return self.candidate_score_branch(candidate_repr).squeeze(-1)

    def interaction(self, candidate_repr: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        if hidden.shape[0] == 1 and candidate_repr.shape[0] != 1:
            hidden = hidden.expand(candidate_repr.shape[0], -1)
        wh = torch.matmul(hidden, self.bilinear_weight.t())
        return (candidate_repr * wh).sum(dim=-1)

    def score(self, candidate_repr: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return self.candidate_score(candidate_repr) + self.interaction(candidate_repr, hidden)

    def aux_score(self, candidate_repr: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        if hidden.shape[0] == 1 and candidate_repr.shape[0] != 1:
            hidden = hidden.expand(candidate_repr.shape[0], -1)
        wh = torch.matmul(hidden, self.aux_bilinear_weight.t())
        return (candidate_repr * wh).sum(dim=-1)

    def forward(self, candidate_features, candidate_mask,
                gold_index_for_update=None, ablate_memory=False, ablate_shuffle=False):
        if candidate_features.dim() == 4:
            candidate_features = candidate_features.squeeze(0)
            candidate_mask = candidate_mask.squeeze(0)
            if gold_index_for_update is not None:
                gold_index_for_update = gold_index_for_update.squeeze(0)

        seq_len, max_cand, _ = candidate_features.shape
        device = candidate_features.device

        h = torch.zeros(1, self.hidden_dim, device=device)
        all_scores = []
        all_interactions = []
        all_aux_scores = []

        for t in range(seq_len):
            if ablate_memory:
                h = torch.zeros(1, self.hidden_dim, device=device)

            feats_t = candidate_features[t]
            mask_t = candidate_mask[t]

            cand_vecs = self.encode_candidates(feats_t)
            h_expanded = h.expand(max_cand, -1)
            interactions_t = self.interaction(cand_vecs, h_expanded)
            scores_t = self.score(cand_vecs, h_expanded)
            scores_t = scores_t.masked_fill(~mask_t, float('-inf'))

            # Compute auxiliary score (cᵀ W_aux h) BEFORE h is updated with the current gold/predicted speaker
            aux_scores_t = self.aux_score(cand_vecs, h_expanded)
            aux_scores_t = aux_scores_t.masked_fill(~mask_t, float('-inf'))

            all_scores.append(scores_t)
            all_interactions.append(interactions_t.masked_fill(~mask_t, 0.0))
            all_aux_scores.append(aux_scores_t)

            if gold_index_for_update is not None:
                spk_idx = gold_index_for_update[t].item()
            else:
                spk_idx = torch.argmax(scores_t).item()

            if ablate_shuffle:
                valid_indices = torch.nonzero(mask_t, as_tuple=False).flatten()
                if len(valid_indices) > 0:
                    spk_idx = valid_indices[torch.randint(0, len(valid_indices), (1,), device=device)].item()
                else:
                    spk_idx = 0

            spk_vec = cand_vecs[spk_idx].unsqueeze(0)
            h = self.gru_cell(spk_vec, h)

        scores_stacked = torch.stack(all_scores, dim=0).unsqueeze(0)
        interactions_stacked = torch.stack(all_interactions, dim=0).unsqueeze(0)
        aux_scores_stacked = torch.stack(all_aux_scores, dim=0).unsqueeze(0)

        return scores_stacked, interactions_stacked, aux_scores_stacked


def compute_masked_previous_speaker_ce(aux_scores, candidate_ids, gold_speaker_id, mask):
    """
    Computes masked candidate-level cross-entropy loss for the auxiliary task.
    aux_scores: [1, seq_len, max_candidates]
    candidate_ids: [1, seq_len, max_candidates] or [seq_len, max_candidates]
    gold_speaker_id: [1, seq_len] or [seq_len]
    mask: [1, seq_len, max_candidates] or [seq_len, max_candidates]
    """
    if aux_scores.dim() == 3:
        aux_scores = aux_scores.squeeze(0)
    if candidate_ids.dim() == 3:
        candidate_ids = candidate_ids.squeeze(0)
    if gold_speaker_id.dim() == 2:
        gold_speaker_id = gold_speaker_id.squeeze(0)
    if mask.dim() == 3:
        mask = mask.squeeze(0)

    seq_len, max_cand = aux_scores.shape
    device = aux_scores.device

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    total_loss = torch.tensor(0.0, device=device)
    valid_count = 0

    for t in range(seq_len):
        if t == 0:
            continue
        
        prev_gold_spk = gold_speaker_id[t - 1].item()
        if prev_gold_spk <= 0:
            continue

        cids_t = candidate_ids[t]
        mask_t = mask[t]

        # Check if previous gold speaker is present in current candidate set and valid
        idx_matches = (cids_t == prev_gold_spk) & mask_t
        matching_indices = torch.nonzero(idx_matches, as_tuple=False).flatten()

        if len(matching_indices) == 1:
            target_idx = matching_indices[0]
            scores_t = aux_scores[t].unsqueeze(0) # [1, max_candidates]
            target_t = target_idx.unsqueeze(0)    # [1]
            
            loss_t = loss_fn(scores_t, target_t)
            total_loss += loss_t.squeeze(0)
            valid_count += 1

    if valid_count > 0:
        return total_loss / valid_count
    return torch.tensor(0.0, device=device)
