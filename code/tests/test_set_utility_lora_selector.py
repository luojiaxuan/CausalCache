from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from causalcache.set_utility_lora_selector import (
    FullSequenceBoundaryEntity,
    SelectorSideLoRAMarginalPredictor,
)
from causalcache.set_utility_selector_branch import LoRALinear
from causalcache.set_utility_token_models import (
    TokenConditionalMarginalPredictor,
    TokenUtilityModelConfig,
)


class _FakeAttention(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, values):
        mixed = self.q_proj(values) + self.k_proj(values) + self.v_proj(values)
        return self.o_proj(torch.tanh(mixed))


class _FakeLayer(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = _FakeAttention(hidden_size)

    def forward(self, values):
        return values + self.self_attn(values)


class _FakePrunedTextModel(torch.nn.Module):
    def __init__(self, hidden_size: int, layer_count: int = 4) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            _FakeLayer(hidden_size) for _ in range(layer_count)
        )
        self.output_norm = torch.nn.LayerNorm(hidden_size)
        self.forward_calls: list[tuple[int, int]] = []

    def forward(
        self,
        *,
        input_ids,
        inputs_embeds,
        attention_mask,
        position_ids,
        use_cache,
        return_dict,
    ):
        assert input_ids is None
        assert attention_mask.shape == inputs_embeds.shape[:2]
        assert position_ids.shape == (3, *inputs_embeds.shape[:2])
        assert use_cache is False
        assert return_dict is True
        self.forward_calls.append(tuple(inputs_embeds.shape[:2]))
        values = inputs_embeds
        for layer in self.layers:
            values = layer(values)
        return SimpleNamespace(last_hidden_state=self.output_norm(values))


def _predictor(hidden_size: int = 8) -> TokenConditionalMarginalPredictor:
    return TokenConditionalMarginalPredictor(
        TokenUtilityModelConfig(
            family="set_transformer",
            source_hidden_size=hidden_size,
            numeric_feature_size=3,
            hidden_size=8,
            latent_count=2,
            resampler_layers=1,
            set_layers=1,
            num_heads=2,
            dropout=0.0,
            preserve_entity_latents=True,
        )
    )


def _entity(length: int, *, seed: int, batched_cache: bool = False):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(length, 8, generator=generator)
    input_ids = torch.arange(length, dtype=torch.long)
    attention_mask = torch.ones(length, dtype=torch.long)
    position_ids = torch.arange(length, dtype=torch.long).repeat(3, 1)
    visual_positions = torch.tensor([length - 2, length - 1], dtype=torch.long)
    text_positions = torch.arange(1, min(3, length - 2), dtype=torch.long)
    if batched_cache:
        hidden = hidden.unsqueeze(0)
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        position_ids = position_ids.unsqueeze(1)
    return FullSequenceBoundaryEntity(
        boundary_hidden_state=hidden,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_positions=visual_positions,
        text_positions=text_positions,
    )


def _base_outputs(model, entities):
    outputs = []
    with torch.no_grad():
        for entity in entities:
            hidden = entity.boundary_hidden_state
            if hidden.ndim == 2:
                hidden = hidden.unsqueeze(0)
            attention_mask = entity.attention_mask
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            position_ids = entity.position_ids
            if position_ids.ndim == 2:
                position_ids = position_ids.unsqueeze(1)
            final = model(
                input_ids=None,
                inputs_embeds=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[0]
            outputs.append(
                (
                    final.index_select(0, entity.visual_positions),
                    final.index_select(0, entity.text_positions),
                )
            )
    return outputs


def test_zero_initialized_top4_lora_preserves_final_token_interface() -> None:
    top_model = _FakePrunedTextModel(8)
    query_entities = (_entity(6, seed=1, batched_cache=True), _entity(5, seed=2))
    event_entities = (
        (_entity(6, seed=3), _entity(5, seed=4)),
        (_entity(6, seed=5), None),
    )
    flat_entities = (
        *query_entities,
        event_entities[0][0],
        event_entities[0][1],
        event_entities[1][0],
    )
    expected = _base_outputs(top_model, flat_entities)
    top_model.forward_calls.clear()
    model = SelectorSideLoRAMarginalPredictor(
        top_language_model=top_model,
        predictor=_predictor(),
        lora_rank=2,
        lora_alpha=4,
        entity_microbatch_size=2,
    )
    tokens = model.encode_predictor_tokens(
        query_entities=query_entities,
        event_entities=event_entities,
        event_mask=torch.tensor([[True, True], [True, False]]),
    )

    assert top_model.forward_calls == [(2, 5), (2, 6), (1, 6)]
    assert torch.allclose(tokens.query_visual_tokens[0], expected[0][0], atol=1e-6)
    assert torch.allclose(tokens.query_visual_tokens[1], expected[1][0], atol=1e-6)
    assert torch.allclose(
        tokens.query_text_tokens[0, : expected[0][1].shape[0]],
        expected[0][1],
        atol=1e-6,
    )
    assert torch.allclose(tokens.event_visual_tokens[0, 0], expected[2][0], atol=1e-6)
    assert torch.allclose(tokens.event_visual_tokens[0, 1], expected[3][0], atol=1e-6)
    assert torch.allclose(tokens.event_visual_tokens[1, 0], expected[4][0], atol=1e-6)
    assert not bool(tokens.event_visual_mask[1, 1].any())
    assert not bool(tokens.event_text_mask[1, 1].any())
    assert all(
        isinstance(module, LoRALinear)
        for layer in top_model.layers
        for module in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
        )
    )


def test_lora_composite_forwards_gradients_and_exposes_disjoint_groups() -> None:
    model = SelectorSideLoRAMarginalPredictor(
        top_language_model=_FakePrunedTextModel(8),
        predictor=_predictor(),
        lora_rank=2,
        lora_alpha=4,
        entity_microbatch_size=4,
    )
    scores = model(
        query_entities=(_entity(6, seed=11),),
        event_entities=((_entity(6, seed=12), _entity(5, seed=13)),),
        event_numeric_features=torch.randn(1, 2, 3),
        event_mask=torch.tensor([[True, True]]),
        selected_masks=torch.tensor([[[False, False], [True, False]]]),
    )
    assert scores.shape == (1, 2, 3)
    scores.sum().backward()
    lora_named = model.named_lora_parameters()
    head_named = model.named_head_parameters()
    assert len(lora_named) == 4 * 4 * 2
    assert any(
        "lora_b" in name and parameter.grad is not None
        for name, parameter in lora_named
    )
    assert any(parameter.grad is not None for _, parameter in head_named)
    groups = model.optimizer_parameter_groups(
        lora_learning_rate=2e-5,
        head_learning_rate=2e-4,
        weight_decay=0.01,
    )
    assert [group["name"] for group in groups] == ["selector_lora", "selector_head"]
    assert [group["lr"] for group in groups] == [2e-5, 2e-4]
    assert not (
        {id(value) for value in groups[0]["params"]}
        & {id(value) for value in groups[1]["params"]}
    )
    assert all(
        not parameter.requires_grad
        for layer in model.top_language_model.layers
        for module in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
        )
        for parameter in module.base.parameters()
    )


def test_lora_composite_restores_only_direct_head_trainability_mask() -> None:
    model = SelectorSideLoRAMarginalPredictor(
        top_language_model=_FakePrunedTextModel(8),
        predictor=_predictor(),
        lora_rank=2,
        lora_alpha=4,
    )
    default_trainable = {
        name
        for name, parameter in model.predictor.named_parameters()
        if parameter.requires_grad
    }
    default_frozen = {
        name
        for name, parameter in model.predictor.named_parameters()
        if not parameter.requires_grad
    }
    assert default_trainable and default_frozen
    model.set_head_trainable(False)
    assert not any(
        parameter.requires_grad for parameter in model.predictor.parameters()
    )
    model.set_head_trainable(True)
    parameters = dict(model.predictor.named_parameters())
    assert {
        name for name, parameter in parameters.items() if parameter.requires_grad
    } == default_trainable
    assert all(not parameters[name].requires_grad for name in default_frozen)


def test_lora_composite_rejects_non_top4_and_inconsistent_padding() -> None:
    with pytest.raises(ValueError, match="top four"):
        SelectorSideLoRAMarginalPredictor(
            top_language_model=_FakePrunedTextModel(8, layer_count=3),
            predictor=_predictor(),
            lora_rank=2,
            lora_alpha=4,
        )

    model = SelectorSideLoRAMarginalPredictor(
        top_language_model=_FakePrunedTextModel(8),
        predictor=_predictor(),
        lora_rank=2,
        lora_alpha=4,
    )
    with pytest.raises(ValueError, match="padded event"):
        model.encode_predictor_tokens(
            query_entities=(_entity(6, seed=21),),
            event_entities=((_entity(6, seed=22), _entity(6, seed=23)),),
            event_mask=torch.tensor([[True, False]]),
        )
