import torch

from src.neural.models import (
    EXP026ABilinearSpeakerGRU,
    EXP026ACandidateOnlyScorer,
    RelationalSpeakerGRU,
)


APPROVED_FEATURES = [
    "candidate_in_quote_chain",
    "candidate_is_attributed_speaker",
    "candidate_is_explicit_mention",
    "candidate_is_recent_mention",
    "chain_recency",
    "conversation_length",
    "conversation_turn_index",
    "discourse_context_length",
    "discourse_dialogue_position",
    "lexical_has_exclamation",
    "lexical_has_question_mark",
    "lexical_quote_length_chars",
    "lexical_quote_length_tokens",
    "nearest_coref_dist",
    "recent_mention_count",
]


def encoder_signature(model):
    return [
        (type(layer).__name__, tuple(layer.weight.shape) if hasattr(layer, "weight") else None)
        for layer in model.candidate_encoder
    ]


def gru_signature(model):
    gru = model.gru_cell
    return {
        "input_size": gru.input_size,
        "hidden_size": gru.hidden_size,
        "bias": gru.bias,
    }


def test_zero_hidden_reduces_to_candidate_only_score():
    torch.manual_seed(1)
    model = EXP026ABilinearSpeakerGRU(feature_dim=15, hidden_dim=64)
    candidate_repr = torch.randn(7, 64)
    hidden = torch.zeros(7, 64)

    scores_zero_state = model.score(candidate_repr, hidden)
    scores_candidate_only = model.candidate_score(candidate_repr)

    assert torch.allclose(
        scores_zero_state,
        scores_candidate_only,
        atol=1e-6,
    )


def test_zero_hidden_removes_bilinear_memory_contribution():
    torch.manual_seed(1)
    model = EXP026ABilinearSpeakerGRU(feature_dim=15, hidden_dim=64)
    candidate_repr = torch.randn(7, 64)
    hidden = torch.zeros(7, 64)
    candidate_scores = model.candidate_score(candidate_repr)

    assert torch.allclose(
        model.interaction(candidate_repr, hidden),
        torch.zeros_like(candidate_scores),
        atol=1e-6,
    )


def test_bilinear_interaction_branch_has_only_w_parameter():
    model = EXP026ABilinearSpeakerGRU(feature_dim=15, hidden_dim=64)
    interaction_params = [
        name for name, _ in model.named_parameters()
        if "bilinear" in name.lower()
    ]

    assert interaction_params == ["bilinear_weight"]
    assert model.bilinear_weight.shape == (64, 64)
    assert model.bilinear_weight.numel() == 4096


def test_exp026a_variant_feature_and_architecture_equivalence():
    variant_a = EXP026ACandidateOnlyScorer(feature_dim=15, hidden_dim=64)
    variant_b = RelationalSpeakerGRU(feature_dim=15, hidden_dim=64)
    variant_c = EXP026ABilinearSpeakerGRU(feature_dim=15, hidden_dim=64)

    variant_a.feature_names = APPROVED_FEATURES
    variant_b.feature_names = APPROVED_FEATURES
    variant_c.feature_names = APPROVED_FEATURES

    assert variant_a.feature_names == variant_b.feature_names
    assert variant_b.feature_names == variant_c.feature_names
    assert encoder_signature(variant_a) == encoder_signature(variant_c)
    assert encoder_signature(variant_b) == encoder_signature(variant_c)
    assert gru_signature(variant_b) == gru_signature(variant_c)


def test_forward_outputs_preserve_quote_and_candidate_alignment():
    torch.manual_seed(1)
    variant_a = EXP026ACandidateOnlyScorer(feature_dim=15, hidden_dim=64)
    variant_b = RelationalSpeakerGRU(feature_dim=15, hidden_dim=64)
    variant_c = EXP026ABilinearSpeakerGRU(feature_dim=15, hidden_dim=64)

    quote_ids_a = ["novel_1", "novel_2", "novel_3"]
    quote_ids_b = list(quote_ids_a)
    quote_ids_c = list(quote_ids_a)

    candidate_features = torch.randn(1, 3, 5, 15)
    candidate_masks_a = torch.ones(1, 3, 5, dtype=torch.bool)
    candidate_masks_b = candidate_masks_a.clone()
    candidate_masks_c = candidate_masks_a.clone()
    candidate_masks_a[:, :, -1] = False
    candidate_masks_b[:, :, -1] = False
    candidate_masks_c[:, :, -1] = False

    scores_a, _ = variant_a(candidate_features, candidate_masks_a)
    scores_b, _ = variant_b(candidate_features, candidate_masks_b)
    scores_c, interactions_c = variant_c(candidate_features, candidate_masks_c)

    assert quote_ids_a == quote_ids_b == quote_ids_c
    assert torch.equal(candidate_masks_a, candidate_masks_b)
    assert torch.equal(candidate_masks_b, candidate_masks_c)
    assert scores_a.shape == scores_b.shape == scores_c.shape == (1, 3, 5)
    assert interactions_c.shape == (1, 3, 5)
    assert torch.isneginf(scores_a[:, :, -1]).all()
    assert torch.isneginf(scores_b[:, :, -1]).all()
    assert torch.isneginf(scores_c[:, :, -1]).all()
