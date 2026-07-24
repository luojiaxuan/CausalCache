"""HGKV counterfactual-readout extractor unit tests on a tiny stub model.

# note (luojiaxuan): 不触真实模型/数据。覆盖:capture 的 mask 对齐(捕获行 ==
# 指定历史位置的冻结投影输出)、lora_delta 与 (alpha/rank)·B·A·x 及 wrapper 门控
# 前向逐位一致、lora_b 全零时 Δr=0/cos=1/质量不变、单历史 token 的精确值(检验
# GQA 头展开与拼接布局)、以及断点续跑幂等键与 singleton 行过滤。
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from causalcache.policy.history_adapter_context import (
    HistoryAdapterContext,
    history_adapter_scope,
)
from causalcache.policy.history_gated_lora import inject_history_gated_kv
from scripts.extract_hgkv_readout_features import (
    build_layer_handles,
    capture_layer_states,
    extract_sample_features,
    is_singleton_row,
    layer_readout_feature,
    load_done_keys,
    lora_delta,
    sample_resume_key,
    singleton_restored_id,
)

_HIDDEN = 8
_HEAD_DIM = 2
_Q_OUT = 8  # 4 query heads
_KV_OUT = 4  # 2 kv heads -> GQA group 2
_LAYER_TOTAL = 3
_LAYER_COUNT = 2
_SEQ = 10
_HIST = (2, 3, 4)
_QUERY_POS = 8


class _StubAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(_HIDDEN, _Q_OUT)
        self.k_proj = torch.nn.Linear(_HIDDEN, _KV_OUT)
        self.v_proj = torch.nn.Linear(_HIDDEN, _KV_OUT)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # note (luojiaxuan): 触发三个投影(捕获 hook 靠 k/v 前向)但不改 hidden,
        # 让每层的注意力输入都等于 embed 输出,便于逐位断言。
        _ = self.q_proj(hidden), self.k_proj(hidden), self.v_proj(hidden)
        return hidden


class _StubBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _StubAttention()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.self_attn(hidden)


class _StubModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.ModuleDict(
            {
                "layers": torch.nn.ModuleList(
                    _StubBlock() for _ in range(_LAYER_TOTAL)
                )
            }
        )

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq = int(input_ids.shape[1])
        hidden = torch.zeros((1, seq, _HIDDEN))
        for position in range(seq):
            hidden[0, position, position % _HIDDEN] = 1.0 + position
            hidden[0, position, (position + 3) % _HIDDEN] = 0.5
        return hidden

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        hidden = self.embed(input_ids)
        for block in self.model["layers"]:
            hidden = block(hidden)
        return hidden


def _make_model(seed: int = 0):
    torch.manual_seed(seed)
    model = _StubModel()
    wrapped = inject_history_gated_kv(
        model, layer_count=_LAYER_COUNT, rank=2, alpha=4
    )
    layers = build_layer_handles(model, wrapped)
    return model, wrapped, layers


def _encoded() -> dict[str, torch.Tensor]:
    return {"input_ids": torch.arange(_SEQ).unsqueeze(0)}


def _hist_mask() -> torch.Tensor:
    mask = torch.zeros((1, _SEQ), dtype=torch.bool)
    mask[0, list(_HIST)] = True
    return mask


def test_layer_handles_are_last_layers_in_ascending_order() -> None:
    model, _, layers = _make_model()
    assert [layer.layer_index for layer in layers] == [1, 2]
    for layer in layers:
        block = model.model["layers"][layer.layer_index]
        assert layer.q_proj is block.self_attn.q_proj
        assert layer.k_wrap.module is block.self_attn.k_proj
        assert layer.v_wrap.module is block.self_attn.v_proj


def test_capture_alignment_matches_history_mask_positions() -> None:
    model, _, layers = _make_model()
    hist_idx = _hist_mask()[0].nonzero(as_tuple=False).squeeze(-1)
    captures = capture_layer_states(
        model, _encoded(), layers=layers, hist_idx=hist_idx, query_pos=_QUERY_POS
    )
    hidden = model.embed(_encoded()["input_ids"])[0]
    for layer in layers:
        slot = captures[layer.layer_index]
        k_proj = layer.k_wrap.module
        v_proj = layer.v_wrap.module
        assert torch.allclose(slot["hidden_query"], hidden[_QUERY_POS])
        assert torch.allclose(slot["hidden_hist"], hidden[list(_HIST)])
        assert torch.allclose(slot["k_hist"], k_proj(hidden[list(_HIST)]))
        assert torch.allclose(slot["v_hist"], v_proj(hidden[list(_HIST)]))
        assert slot["k_prompt"].shape[0] == _QUERY_POS + 1
        assert torch.allclose(
            slot["k_prompt"], k_proj(hidden[: _QUERY_POS + 1])
        )


def test_lora_delta_matches_scaled_ba_product_and_wrapper_forward() -> None:
    model, wrapped, layers = _make_model()
    k_wrap = layers[0].k_wrap
    with torch.no_grad():
        k_wrap.lora_b.copy_(torch.randn_like(k_wrap.lora_b))
    x = torch.randn(len(_HIST), _HIDDEN)
    manual = (x @ k_wrap.lora_a.T @ k_wrap.lora_b.T) * (4 / 2)
    assert torch.allclose(lora_delta(k_wrap, x), manual, atol=1e-6)

    # note (luojiaxuan): 与 wrapper 真实门控前向逐位对照——装上 mask context 后
    # k_proj 输出在历史位置应恰为 K0 + lora_delta,非历史位置保持冻结输出。
    hidden = model.embed(_encoded()["input_ids"])
    context = HistoryAdapterContext(
        history_token_mask=_hist_mask(),
        history_present=True,
        image_roles=("history", "current"),
    )
    clean = k_wrap.module(hidden[0])
    with history_adapter_scope(context):
        gated = k_wrap.module(hidden)[0]
    expected = clean.clone()
    expected[list(_HIST)] += lora_delta(k_wrap, hidden[0, list(_HIST)])
    assert torch.allclose(gated, expected, atol=1e-6)


def test_zero_lora_b_gives_zero_readout_delta() -> None:
    model, _, layers = _make_model()
    feature, scalars = extract_sample_features(
        model,
        _encoded(),
        history_mask=_hist_mask(),
        query_pos=_QUERY_POS,
        layers=layers,
        head_dim=_HEAD_DIM,
    )
    n_q_heads = _Q_OUT // _HEAD_DIM
    assert len(feature) == _LAYER_COUNT * (_HEAD_DIM + n_q_heads)
    assert all(value == 0.0 for value in feature)
    assert set(scalars) == {"1", "2"}
    for layer_scalars in scalars.values():
        assert layer_scalars["dr_norm"] == 0.0
        assert layer_scalars["cos_r0_rhg"] == pytest.approx(1.0)
        assert layer_scalars["hist_mass_base"] == pytest.approx(
            layer_scalars["hist_mass_adapted"]
        )
        assert 0.0 < layer_scalars["hist_mass_base"] < 1.0


def test_nonzero_lora_b_moves_readout_and_history_mass() -> None:
    model, wrapped, layers = _make_model(seed=1)
    with torch.no_grad():
        for wrap in wrapped.values():
            wrap.lora_b.copy_(torch.randn_like(wrap.lora_b))
    feature, scalars = extract_sample_features(
        model,
        _encoded(),
        history_mask=_hist_mask(),
        query_pos=_QUERY_POS,
        layers=layers,
        head_dim=_HEAD_DIM,
    )
    assert any(value != 0.0 for value in feature)
    for layer_scalars in scalars.values():
        assert layer_scalars["dr_norm"] > 0.0
        assert (
            layer_scalars["hist_mass_base"]
            != layer_scalars["hist_mass_adapted"]
        )


def test_flat_feature_chunks_follow_ascending_layer_order() -> None:
    model, wrapped, layers = _make_model(seed=2)
    with torch.no_grad():
        for name, wrap in wrapped.items():
            layer_index = int(name.split(".layers.", 1)[1].split(".", 1)[0])
            wrap.lora_b.fill_(0.01 * (layer_index + 1))
    flat, _ = extract_sample_features(
        model,
        _encoded(),
        history_mask=_hist_mask(),
        query_pos=_QUERY_POS,
        layers=layers,
        head_dim=_HEAD_DIM,
    )
    chunk_width = _HEAD_DIM + _Q_OUT // _HEAD_DIM
    assert len(flat) == len(layers) * chunk_width

    hist_idx = _hist_mask()[0].nonzero(as_tuple=False).squeeze(-1)
    captures = capture_layer_states(
        model,
        _encoded(),
        layers=layers,
        hist_idx=hist_idx,
        query_pos=_QUERY_POS,
    )
    expected = []
    for layer in layers:
        slot = captures[layer.layer_index]
        with torch.inference_mode():
            q = layer.q_proj(
                slot["hidden_query"].to(layer.q_proj.weight.dtype)
            ).float()
            feature, _ = layer_readout_feature(
                q=q,
                k0=slot["k_hist"],
                v0=slot["v_hist"],
                dk=lora_delta(layer.k_wrap, slot["hidden_hist"]),
                dv=lora_delta(layer.v_wrap, slot["hidden_hist"]),
                k_prompt=slot["k_prompt"],
                hist_idx=hist_idx,
                head_dim=_HEAD_DIM,
            )
        expected.extend(float(f"{value:.6g}") for value in feature.tolist())
    assert flat == expected


def test_single_history_token_exact_values_and_gqa_expansion() -> None:
    # note (luojiaxuan): T=1 时 softmax 恒为 1,Δr 每个 query 头精确等于对应
    # kv 头的 dv——直接检验 repeat_interleave 头展开与 [mean‖per-head norm]
    # 拼接布局;质量项与独立实现的全前缀 softmax 对照。
    torch.manual_seed(3)
    q = torch.randn(_Q_OUT)
    k_prompt = torch.randn(5, _KV_OUT)
    hist_idx = torch.tensor([2])
    k0 = k_prompt[hist_idx]
    v0 = torch.randn(1, _KV_OUT)
    dk = torch.randn(1, _KV_OUT)
    dv = torch.randn(1, _KV_OUT)
    feature, scalars = layer_readout_feature(
        q=q,
        k0=k0,
        v0=v0,
        dk=dk,
        dv=dv,
        k_prompt=k_prompt,
        hist_idx=hist_idx,
        head_dim=_HEAD_DIM,
    )
    dv_heads = dv.view(2, _HEAD_DIM)
    expected_mean = dv_heads.mean(dim=0)
    expected_norms = torch.tensor(
        [
            float(dv_heads[0].norm()),
            float(dv_heads[0].norm()),
            float(dv_heads[1].norm()),
            float(dv_heads[1].norm()),
        ]
    )
    assert torch.allclose(feature[:_HEAD_DIM], expected_mean, atol=1e-6)
    assert torch.allclose(feature[_HEAD_DIM:], expected_norms, atol=1e-6)
    assert scalars["dr_norm"] == pytest.approx(
        float(torch.cat([dv_heads[0], dv_heads[0], dv_heads[1], dv_heads[1]]).norm()),
        rel=1e-5,
    )

    qh = q.view(4, _HEAD_DIM)
    kp = k_prompt.view(5, 2, _HEAD_DIM).repeat_interleave(2, dim=1)
    base_logits = torch.einsum("hd,phd->hp", qh, kp) / (_HEAD_DIM**0.5)
    adapted_logits = base_logits.clone()
    k1 = (k0 + dk).view(1, 2, _HEAD_DIM).repeat_interleave(2, dim=1)
    adapted_logits[:, 2] = (
        torch.einsum("hd,hd->h", qh, k1[0]) / (_HEAD_DIM**0.5)
    )
    expected_mass0 = float(
        torch.softmax(base_logits, dim=-1)[:, 2].mean()
    )
    expected_mass1 = float(
        torch.softmax(adapted_logits, dim=-1)[:, 2].mean()
    )
    assert scalars["hist_mass_base"] == pytest.approx(expected_mass0, rel=1e-5)
    assert scalars["hist_mass_adapted"] == pytest.approx(expected_mass1, rel=1e-5)


def test_history_tokens_must_precede_query_position() -> None:
    model, _, layers = _make_model()
    mask = torch.zeros((1, _SEQ), dtype=torch.bool)
    mask[0, _QUERY_POS] = True
    with pytest.raises(ValueError, match="precede"):
        extract_sample_features(
            model,
            _encoded(),
            history_mask=mask,
            query_pos=_QUERY_POS,
            layers=layers,
            head_dim=_HEAD_DIM,
        )


def test_singleton_row_filter_and_resume_key() -> None:
    singleton = {
        "pair_group": "ep0:7",
        "variant": "singleton",
        "singleton_event_step_id": 5,
        "memory_config": {"restored_event_step_ids": [5]},
    }
    assert is_singleton_row(singleton)
    assert is_singleton_row({"variant": "single"})
    assert not is_singleton_row({"variant": "b0"})
    assert not is_singleton_row({"variant": "correct"})
    assert not is_singleton_row({})
    assert sample_resume_key(singleton) == ("ep0:7", 5)
    fallback = {
        "pair_group": "ep0:7",
        "variant": "singleton",
        "memory_config": {"restored_event_step_ids": [9]},
    }
    assert singleton_restored_id(fallback) == 9
    with pytest.raises(ValueError, match="expected exactly 1"):
        singleton_restored_id(
            {
                "pair_group": "ep0:7",
                "memory_config": {"restored_event_step_ids": [1, 2]},
            }
        )
    with pytest.raises(ValueError, match="mismatch"):
        singleton_restored_id(
            {
                "pair_group": "ep0:7",
                "singleton_event_step_id": 4,
                "memory_config": {"restored_event_step_ids": [5]},
            }
        )


def test_resume_skip_reads_existing_output(tmp_path) -> None:
    output = tmp_path / "features.jsonl"
    rows = [
        {"pair_group": "ep0:7", "singleton_event_step_id": 5},
        {"pair_group": "ep1:3", "singleton_event_step_id": 2},
    ]
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.write("not-json\n")
        handle.write(json.dumps({"pair_group": "ep2:1"}) + "\n")
    done = load_done_keys(output)
    assert done == {("ep0:7", 5), ("ep1:3", 2)}
    done_sample = {
        "pair_group": "ep0:7",
        "variant": "singleton",
        "singleton_event_step_id": 5,
        "memory_config": {"restored_event_step_ids": [5]},
    }
    pending_sample = {
        "pair_group": "ep0:7",
        "variant": "singleton",
        "singleton_event_step_id": 6,
        "memory_config": {"restored_event_step_ids": [6]},
    }
    assert sample_resume_key(done_sample) in done
    assert sample_resume_key(pending_sample) not in done
    assert load_done_keys(tmp_path / "missing.jsonl") == set()
