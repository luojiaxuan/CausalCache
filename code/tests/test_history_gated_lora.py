from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from causalcache.policy.history_adapter_context import (
    HistoryAdapterContext,
    get_history_adapter_context,
    history_adapter_scope,
)
from causalcache.policy.history_gated_lora import (
    history_gated_state_dict,
    inject_history_gated_kv,
    load_history_gated_state_dict,
)

_HIDDEN = 8
_LAYER_TOTAL = 4
_LAYER_COUNT = 2


class _FakeAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(_HIDDEN, _HIDDEN)
        self.k_proj = torch.nn.Linear(_HIDDEN, _HIDDEN)
        self.v_proj = torch.nn.Linear(_HIDDEN, _HIDDEN)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.k_proj(hidden) + self.v_proj(hidden) + self.q_proj(hidden)


class _FakeLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttention()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.self_attn(hidden)


class _FakeVisual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k_proj = torch.nn.Linear(_HIDDEN, _HIDDEN)


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _FakeVisual()
        self.language_model = torch.nn.ModuleDict(
            {
                "layers": torch.nn.ModuleList(
                    _FakeLayer() for _ in range(_LAYER_TOTAL)
                )
            }
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.language_model["layers"]:
            hidden = layer(hidden)
        return hidden


def _frozen_model(seed: int = 0) -> _FakeModel:
    torch.manual_seed(seed)
    model = _FakeModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _inject_nonzero(model: _FakeModel) -> dict:
    # note (luojiaxuan): lora_b 注入后置为非零,让 residual 一旦被错误执行就
    # 必然改变输出,使 bypass 断言真正具备判别力。
    wrapped = inject_history_gated_kv(
        model, layer_count=_LAYER_COUNT, rank=2, alpha=4
    )
    with torch.no_grad():
        for lora in wrapped.values():
            lora.lora_b.uniform_(-0.5, 0.5)
    return wrapped


def _context(mask: torch.Tensor) -> HistoryAdapterContext:
    present = bool(mask.any())
    return HistoryAdapterContext(
        history_token_mask=mask,
        history_present=present,
        image_roles=("history", "current") if present else ("current",),
    )


def test_injection_targets_last_layers_k_and_v_only() -> None:
    model = _frozen_model()
    wrapped = _inject_nonzero(model)
    assert set(wrapped) == {
        f"language_model.layers.{index}.self_attn.{projection}"
        for index in (_LAYER_TOTAL - _LAYER_COUNT, _LAYER_TOTAL - 1)
        for projection in ("k_proj", "v_proj")
    }


def test_injection_rejects_oversized_layer_count() -> None:
    with pytest.raises(RuntimeError):
        inject_history_gated_kv(_frozen_model(), layer_count=_LAYER_TOTAL + 1)


def test_bypass_without_context_is_bitwise_equal() -> None:
    model = _frozen_model()
    prompt = torch.randn(1, 5, _HIDDEN)
    with torch.no_grad():
        baseline = model(prompt)
    _inject_nonzero(model)
    assert get_history_adapter_context() is None
    with torch.no_grad():
        gated = model(prompt)
    assert torch.equal(gated, baseline)


def test_bypass_with_all_false_mask_is_bitwise_equal() -> None:
    model = _frozen_model()
    prompt = torch.randn(1, 5, _HIDDEN)
    with torch.no_grad():
        baseline = model(prompt)
    _inject_nonzero(model)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    with history_adapter_scope(_context(mask)), torch.no_grad():
        gated = model(prompt)
    assert torch.equal(gated, baseline)
    assert get_history_adapter_context() is None


def test_partial_mask_changes_only_masked_positions() -> None:
    model = _frozen_model()
    wrapped = _inject_nonzero(model)
    name = f"language_model.layers.{_LAYER_TOTAL - 1}.self_attn.k_proj"
    lora = wrapped[name]
    module = model.language_model["layers"][_LAYER_TOTAL - 1].self_attn.k_proj
    prompt = torch.randn(1, 5, _HIDDEN)
    frozen = torch.nn.functional.linear(prompt, module.weight, module.bias)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    mask[0, 1] = True
    mask[0, 3] = True
    with history_adapter_scope(_context(mask)), torch.no_grad():
        gated = module(prompt)
    assert torch.equal(gated[~mask], frozen[~mask])
    assert not torch.equal(gated[mask], frozen[mask])
    expected = frozen + (
        (prompt @ lora.lora_a.T @ lora.lora_b.T) * lora.scaling
    ) * mask.unsqueeze(-1)
    assert torch.allclose(gated, expected, atol=1e-6)


def test_length_mismatch_bypasses_bitwise() -> None:
    model = _frozen_model()
    decode_step = torch.randn(1, 1, _HIDDEN)
    teacher_forced = torch.randn(1, 7, _HIDDEN)
    with torch.no_grad():
        decode_baseline = model(decode_step)
        teacher_baseline = model(teacher_forced)
    _inject_nonzero(model)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    mask[0, 2] = True
    with history_adapter_scope(_context(mask)), torch.no_grad():
        assert torch.equal(model(decode_step), decode_baseline)
        assert torch.equal(model(teacher_forced), teacher_baseline)


def test_state_dict_roundtrip_and_key_inventory() -> None:
    source = _inject_nonzero(_frozen_model(seed=1))
    state = history_gated_state_dict(source)
    assert set(state) == {
        f"{name}.{part}" for name in source for part in ("lora_a", "lora_b")
    }
    target = inject_history_gated_kv(
        _frozen_model(seed=2), layer_count=_LAYER_COUNT, rank=2, alpha=4
    )
    load_history_gated_state_dict(target, state)
    for name, lora in target.items():
        assert torch.equal(lora.lora_a.detach(), state[f"{name}.lora_a"])
        assert torch.equal(lora.lora_b.detach(), state[f"{name}.lora_b"])
    with pytest.raises(ValueError):
        load_history_gated_state_dict(
            target,
            {key: value for key, value in state.items() if key.endswith("lora_a")},
        )
