"""Desktop DiD 语料 ←→ trainer 契约的冻结不变量(交接 §8.2-3)。

# note (luojiaxuan): 三层锁:
#   1. build_desktop_hgkv_corpus 产出的行必须原样通过 trainer 的样本/组校验、
#      训练单元与留出单元构建(split 先经 normalize_desktop_splits);
#   2. osworld_official 编码路径与在线 GUIOwlOSWorldRuntime.generate_raw 逐实参
#      相同(tools=[_TOOL_SPEC]、单会话实参),旧 chat_template 路径一个实参不变;
#      真模型 token-by-token parity 由 audit_osworld_official_trainer_parity.py 执行;
#   3. 量名/必需 gate 按 schema 分派:桌面负臂 WA 的差值量名是 SA_minus_WA,
#      v2/v6 的量名逐字节不变。
"""

from __future__ import annotations

import io
import json

import pytest
import torch
from PIL import Image

import scripts.train_success_sft_lora as trainer
from causalcache.osworld_gui_owl import _TOOL_SPEC
from scripts.build_desktop_hgkv_corpus import build_corpus


STEPS = 8


def tool(action: str, **kwargs):
    return {"name": "computer_use", "arguments": {"action": action, **kwargs}}


def png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def synthetic_record(tmp_path, *, dp_id="dp-1", task_id="traj-1"):
    relpaths = []
    (tmp_path / task_id).mkdir(exist_ok=True)
    for index in range(STEPS):
        relpath = f"{task_id}/obs-{index:03d}.png"
        (tmp_path / relpath).write_bytes(png_bytes((index * 25 % 255, 40, 70)))
        relpaths.append(relpath)
    history = [
        {"step_id": 1, "action": {"type": "click", "x": 10, "y": 10}},
        {"step_id": 2, "action": {"type": "click", "x": 500, "y": 500}},
        {"step_id": 3, "action": {"type": "press", "key": "tab"}},
        {"step_id": 4, "action": {"type": "click", "x": 50, "y": 50}},
        {"step_id": 5, "action": {"type": "click", "x": 500, "y": 500}},
        {"step_id": 6, "action": {"type": "press", "key": "enter"}},
        {"step_id": 7, "action": {"type": "click", "x": 80, "y": 80}},
    ]
    for entry in history:
        entry["osworld_action"] = f"pyautogui.{entry['action']['type']}()"
    return {
        "dp_id": dp_id,
        "task_id": task_id,
        "os": "ubuntu",
        "step": STEPS,
        "screen_size": [1000, 1000],
        "instruction": "demo task",
        "target_tool_call": tool("left_click", coordinate=[500, 500]),
        "target_text": '{"action": "left_click", "coordinate": [500, 500]}',
        "history": history,
        "image_relpaths": relpaths,
    }


def corpus_rows(tmp_path, *, split: str) -> list[dict]:
    rows, _b0, _manifest = build_corpus(
        [synthetic_record(tmp_path)],
        image_root=tmp_path,
        coordinate_tolerance=2,
        min_age=3,
        seed=7,
    )
    return [{**row, "split": split} for row in rows]


def as_group(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    return rows, {row["arm_slot"]: index for index, row in enumerate(rows)}


# ---------------------------------------------------------------------------
# 1. 语料行 ←→ trainer 校验/单元构建
# ---------------------------------------------------------------------------


def test_desktop_rows_pass_trainer_sample_and_group_validation(tmp_path) -> None:
    rows = corpus_rows(tmp_path, split="train")
    for index, row in enumerate(rows):
        trainer.validate_sparse_sample(row, index=index)
    samples, group = as_group(rows)
    assert (
        trainer.validate_sparse_group(
            samples, pair_group=rows[0]["pair_group"], group=group
        )
        == "R0"
    )


def test_desktop_train_units_build_under_did_ra_aware(tmp_path) -> None:
    rows = corpus_rows(tmp_path, split="train")
    units, heldout = trainer.build_sparse_history_units(
        rows, objective_kind=trainer.SPARSE_OBJECTIVE_DID_RA_AWARE
    )
    assert len(units) == 1 and heldout == set()
    kind, group, _ = units[0]
    assert kind == "sparse_group"
    assert sorted(group) == ["R0", "RA", "S0", "SA", "WA"]


def test_desktop_heldout_units_do_not_require_n0(tmp_path) -> None:
    rows = corpus_rows(tmp_path, split="heldout")
    groups = trainer.build_sparse_history_heldout_units(rows)
    assert len(groups) == 1
    (group,) = groups.values()
    assert sorted(group) == ["R0", "RA", "S0", "SA", "WA"]


def test_desktop_deployment_baseline_must_be_r0(tmp_path) -> None:
    rows = corpus_rows(tmp_path, split="train")
    broken = {**rows[0], "deployment_baseline_arm_id": "N0"}
    with pytest.raises(ValueError, match="deployment_baseline_arm_id"):
        trainer.validate_sparse_sample(broken, index=0)


def test_desktop_negative_slot_binds_to_wrong_kind(tmp_path) -> None:
    rows = corpus_rows(tmp_path, split="train")
    negative = next(row for row in rows if row["arm_slot"] == "WA")
    mismatched = {**negative, "negative_kind": "age_matched"}
    with pytest.raises(ValueError, match="disagrees with negative_kind"):
        trainer.validate_sparse_sample(mismatched, index=0)
    foreign = {**negative, "donor_episode": "another-episode"}
    with pytest.raises(ValueError, match="its own episode"):
        trainer.validate_sparse_sample(foreign, index=0)


# ---------------------------------------------------------------------------
# 2. split 归一(train/dev/test → train/heldout)
# ---------------------------------------------------------------------------


def _desktop_stub(split: str, sample_id: str) -> dict:
    return {
        "schema_version": trainer.SPARSE_DESKTOP_SAMPLE_SCHEMA,
        "sample_id": sample_id,
        "split": split,
    }


def test_normalize_desktop_splits_maps_dev_and_drops_test() -> None:
    foreign = {"schema_version": "something.else", "split": "dev"}
    samples = [
        _desktop_stub("train", "g|R0"),
        _desktop_stub("dev", "g|RA"),
        _desktop_stub("test", "g|SA"),
        foreign,
    ]
    kept, counters = trainer.normalize_desktop_splits(samples)
    assert counters == {"desktop_dev_to_heldout": 1, "desktop_test_dropped": 1}
    assert [sample["split"] for sample in kept] == ["train", "heldout", "dev"]
    # 非桌面 schema 的样本必须原对象直通,一个字段都不动
    assert kept[-1] is foreign


def test_normalize_desktop_splits_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="unknown split"):
        trainer.normalize_desktop_splits([_desktop_stub("heldout", "g|R0")])


def test_unnormalized_desktop_split_fails_validation_with_hint(tmp_path) -> None:
    rows = corpus_rows(tmp_path, split="dev")
    with pytest.raises(ValueError, match="normalize_desktop_splits"):
        trainer.validate_sparse_sample(rows[0], index=0)


# ---------------------------------------------------------------------------
# 3. osworld_official 编码路径
# ---------------------------------------------------------------------------


class _StubTokenizer:
    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [11, 12]}


class _StubProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict]] = []
        self.tokenizer = _StubTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {
            "input_ids": torch.tensor([[5, 6, 7]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }


class _StubRuntime:
    def __init__(self) -> None:
        self.processor = _StubProcessor()
        self.device = "cpu"

    def assert_pinned_assistant_prefix(self, input_ids, batch) -> None:
        return None


def test_prompt_encoding_path_dispatch() -> None:
    assert (
        trainer._prompt_encoding_path({"prompt_format": "osworld_official"})
        == "osworld_chat_template"
    )
    assert (
        trainer._prompt_encoding_path({"prompt_format": "sparse_single_turn"})
        == "chat_template"
    )
    with pytest.raises(ValueError, match="unknown prompt_format"):
        trainer._prompt_encoding_path({"prompt_format": "made_up"})


def test_osworld_encode_matches_online_runtime_arguments(tmp_path) -> None:
    rows = corpus_rows(tmp_path, split="train")
    positive = next(row for row in rows if row["arm_slot"] == "SA")
    runtime = _StubRuntime()
    encoded = trainer.encode_sample(
        runtime, positive, dataset_root=tmp_path, torch=torch
    )
    assert encoded is not None
    (messages, kwargs) = runtime.processor.calls[0]
    # 与 GUIOwlOSWorldRuntime.generate_raw 逐实参相同:单会话实参 + tools=[_TOOL_SPEC]
    assert isinstance(messages, list) and messages[0]["role"] == "system"
    assert kwargs["tools"] == [_TOOL_SPEC] and kwargs["tools"][0] is _TOOL_SPEC
    assert kwargs["tokenize"] is True
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["return_dict"] is True
    assert kwargs["return_tensors"] == "pt"
    assert "padding" not in kwargs
    # prompt 段全 -100,目标段是 tokenizer 给的两枚 token
    assert encoded["input_ids"].tolist() == [[5, 6, 7, 11, 12]]
    assert encoded["labels"].tolist() == [[-100, -100, -100, 11, 12]]


def test_old_chat_template_path_still_has_no_tools_argument(tmp_path) -> None:
    rows = corpus_rows(tmp_path, split="train")
    old_style = {**rows[0], "prompt_format": "sparse_single_turn"}
    runtime = _StubRuntime()
    trainer.encode_sample(runtime, old_style, dataset_root=tmp_path, torch=torch)
    (messages, kwargs) = runtime.processor.calls[0]
    assert "tools" not in kwargs
    # 旧路径包 batch 列表,新路径不包 —— 两条路径不得互相漂移
    assert isinstance(messages, list) and isinstance(messages[0], list)


# ---------------------------------------------------------------------------
# 4. 量名 / gate / objective 的 schema 分派
# ---------------------------------------------------------------------------


def test_diagnostic_keys_follow_the_arm_slot() -> None:
    assert trainer.sparse_diagnostic_keys("wrong", arm_slot="WA") == (
        "SA_minus_WA",
        "wrong_drift_abs",
    )
    # v2/v6 的量名逐字节不变(slot 本就是 SA_neg_<kind>)
    assert trainer.sparse_diagnostic_keys("age_matched") == (
        "SA_minus_SA_neg_age_matched",
        "age_matched_drift_abs",
    )
    assert trainer.sparse_diagnostic_keys(
        "age_matched", arm_slot="SA_neg_age_matched"
    ) == ("SA_minus_SA_neg_age_matched", "age_matched_drift_abs")


def test_desktop_required_gates_and_negative_keys() -> None:
    schema = trainer.SPARSE_DESKTOP_SAMPLE_SCHEMA
    assert trainer.SPARSE_REQUIRED_GATES_BY_SCHEMA[schema] == frozenset(
        {"did_select", "adapter_on_sparse", "adapter_on_recent_abs", "wrong_drift_abs"}
    )
    assert trainer.SPARSE_NEGATIVE_GATE_KEYS_BY_SCHEMA[schema] == frozenset(
        {"SA_minus_WA", "wrong_drift_abs"}
    )


def test_objective_excluded_arms_depend_on_schema() -> None:
    assert trainer.sparse_objective_excluded_arms(
        trainer.SPARSE_OBJECTIVE_DID_RA_AWARE
    ) == ["N0"]
    assert (
        trainer.sparse_objective_excluded_arms(
            trainer.SPARSE_OBJECTIVE_DID_RA_AWARE,
            sample_schema=trainer.SPARSE_DESKTOP_SAMPLE_SCHEMA,
        )
        == []
    )


def desktop_gates_block() -> dict:
    return {
        "gate_schema": "causalcache.sparse_history_gates.v2",
        "reference_arm_id": "R0",
        "main_claim_quantity": "SA_minus_RA",
        "derived_quantities": dict(trainer.SPARSE_DESKTOP_DERIVED_QUANTITIES),
        "must_pass": {
            "did_select": "> 0 @ci_low",
            "adapter_on_sparse": "> 0",
            "adapter_on_recent_abs": "< 0.02",
            "wrong_drift_abs": "< 0.02",
        },
        "drift_definition": "wrong_drift_abs = |mean(A_WA)| on the heldout split",
        "composite_score": "did_select",
        "selection_rule": {
            "filter": "all_must_pass",
            "objective": "maximize",
            "quantity": "composite_score",
            "tie_break": "earliest_checkpoint",
        },
    }


def test_desktop_gates_block_validates() -> None:
    config = {
        "data": {"sample_schema_version": trainer.SPARSE_DESKTOP_SAMPLE_SCHEMA},
        "gates": desktop_gates_block(),
    }
    assert trainer.validate_sparse_gates(config) == config["gates"]


def test_desktop_gates_reject_v2_derived_quantities() -> None:
    gates = desktop_gates_block()
    gates["derived_quantities"] = dict(trainer.SPARSE_DERIVED_QUANTITIES)
    config = {
        "data": {"sample_schema_version": trainer.SPARSE_DESKTOP_SAMPLE_SCHEMA},
        "gates": gates,
    }
    with pytest.raises(ValueError, match="derived_quantities drifted"):
        trainer.validate_sparse_gates(config)


def test_desktop_gates_reject_foreign_negative_quantities() -> None:
    gates = desktop_gates_block()
    gates["must_pass"]["age_matched_drift_abs"] = "< 0.02"
    config = {
        "data": {"sample_schema_version": trainer.SPARSE_DESKTOP_SAMPLE_SCHEMA},
        "gates": gates,
    }
    with pytest.raises(ValueError, match="never\\s+produces those negatives"):
        trainer.validate_sparse_gates(config)


# ---------------------------------------------------------------------------
# 5. DiD 损失在桌面组上的量名与负样本收集
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self, values: dict[str, float]) -> None:
        self.values = values
        self.calls: list[tuple[str, bool, str | None]] = []

    def __call__(
        self,
        slot: str,
        *,
        grad: bool = False,
        adapter_mode: str | None = None,
        backward_weight: float | None = None,
    ) -> float | None:
        self.calls.append((slot, grad, adapter_mode))
        if adapter_mode == "bypass":
            return self.values.get(f"{slot}|bypass", 0.0)
        return self.values.get(slot, 0.0)


# ---------------------------------------------------------------------------
# 5b. 三份桌面训练 config 必须整体通过 trainer 的全部启动校验
# ---------------------------------------------------------------------------

DESKTOP_CONFIG_NAMES = (
    "causalcache_desktop_did_hgkv_v1.json",
    "causalcache_desktop_did_ungated_kv_v1.json",
    "causalcache_desktop_did_full_lora_v1.json",
    "causalcache_desktop_did_hgkv_v3.json",
    "causalcache_desktop_did_ungated_kv_v3.json",
    "causalcache_desktop_did_full_lora_v3.json",
)


@pytest.mark.parametrize("name", DESKTOP_CONFIG_NAMES)
def test_desktop_training_config_passes_all_startup_validators(name) -> None:
    import argparse
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs" / name
    config = trainer.load_config(path)
    adapter_type, options = trainer.adapter_settings(config)
    assert adapter_type in trainer.SPARSE_SUPPORTED_ADAPTER_TYPES
    if adapter_type != "full_policy_lora":
        assert options == {"layer_count": 8, "rank": 8, "alpha": 16}
    assert trainer.sparse_config_sample_schema(config) == (
        trainer.SPARSE_DESKTOP_SAMPLE_SCHEMA
    )
    trainer.validate_sparse_gates(config)
    kind, _objective = trainer.validate_sparse_objective(config)
    assert kind == trainer.SPARSE_OBJECTIVE_DID_RA_AWARE
    controls = trainer.resolve_sparse_training(
        config["training"],
        args=argparse.Namespace(max_steps=0, checkpoint_every_steps=0),
        objective_kind=kind,
    )
    expected = (
        {"max_steps": 300, "checkpoint_every_steps": 50}
        if name.endswith("_v3.json")
        else {"max_steps": 150, "checkpoint_every_steps": 25}
    )
    assert controls == expected
    # 桌面语料没有 label_class 字段,config 不得声明 train_on_label_classes
    assert trainer.resolve_train_on_label_classes(config) is None


@pytest.mark.parametrize("suffix", ["_v1.json", "_v3.json"])
def test_desktop_training_configs_share_the_frozen_protocol(suffix) -> None:
    import json as json_module
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    configs = [
        json_module.loads((root / name).read_text(encoding="utf-8"))
        for name in DESKTOP_CONFIG_NAMES
        if name.endswith(suffix)
    ]
    assert len(configs) == 3
    first = configs[0]
    for other in configs[1:]:
        # §7:同一 split、同一训练组、同 visual tokens、相同 steps 与 cadence
        assert other["training"] == first["training"]
        assert other["gates"] == first["gates"]
        assert other["objective"] == first["objective"]
        assert (
            other["data"]["samples_sha256"] == first["data"]["samples_sha256"]
        )
        assert other["target_effective_visual_tokens_per_image"] == 2560


# ---------------------------------------------------------------------------
# 6. §8.4:Full-layer / ungated-KV 共用 DiD loss 的 bypass 机制
# ---------------------------------------------------------------------------


class _TinyLM(torch.nn.Module):
    def __init__(self, layers: int = 4) -> None:
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.layers = torch.nn.ModuleList()
        for _ in range(layers):
            block = torch.nn.Module()
            block.self_attn = torch.nn.Module()
            block.self_attn.q_proj = torch.nn.Linear(4, 4)
            block.self_attn.k_proj = torch.nn.Linear(4, 4)
            block.self_attn.v_proj = torch.nn.Linear(4, 4)
            self.language_model.layers.append(block)


def test_plain_lora_bypass_scope_restores_the_frozen_output() -> None:
    linear = torch.nn.Linear(4, 3)
    wrapper = trainer.LoRALinear(linear, rank=2, alpha=4, torch=torch)
    with torch.no_grad():
        wrapper.lora_b.fill_(0.37)
    x = torch.randn(2, 4)
    with trainer.plain_lora_bypass_scope(True):
        bypassed = linear(x)
    active = linear(x)
    frozen = torch.nn.functional.linear(x, linear.weight, linear.bias)
    assert torch.equal(bypassed, frozen)
    assert not torch.equal(active, frozen)
    # scope 退出后恢复 active —— 旧 full_policy_lora 路径的缺省行为
    assert trainer._PLAIN_LORA_BYPASS.get() is False


def test_inject_lora_last_layer_count_restricts_to_the_tail() -> None:
    model = _TinyLM(layers=4)
    tail = trainer.inject_lora(
        model,
        rank=2,
        alpha=4,
        target_modules=("k_proj", "v_proj"),
        torch=torch,
        last_layer_count=2,
    )
    assert sorted(tail) == [
        "language_model.layers.2.self_attn.k_proj",
        "language_model.layers.2.self_attn.v_proj",
        "language_model.layers.3.self_attn.k_proj",
        "language_model.layers.3.self_attn.v_proj",
    ]
    everything = trainer.inject_lora(
        _TinyLM(layers=4),
        rank=2,
        alpha=4,
        target_modules=("k_proj", "v_proj"),
        torch=torch,
    )
    assert len(everything) == 8


def test_adapter_settings_accepts_the_matched_ungated_kv_shape() -> None:
    config = {
        "adapter": {
            "adapter_type": "ungated_kv_lora",
            "layer_scope": "last_8",
            "rank": 8,
            "alpha": 16,
        }
    }
    assert trainer.adapter_settings(config) == (
        "ungated_kv_lora",
        {"layer_count": 8, "rank": 8, "alpha": 16},
    )
    with pytest.raises(ValueError, match="layer_scope"):
        trainer.adapter_settings(
            {"adapter": {"adapter_type": "ungated_kv_lora", "layer_scope": "all"}}
        )
    with pytest.raises(ValueError, match="frozen to k/v_proj"):
        trainer.adapter_settings(
            {
                "adapter": {
                    "adapter_type": "ungated_kv_lora",
                    "layer_scope": "last_8",
                    "rank": 8,
                    "alpha": 16,
                    "target_modules": ["q_proj"],
                }
            }
        )


def test_adapter_scope_dispatch_for_plain_adapters() -> None:
    for adapter_type in ("full_policy_lora", "ungated_kv_lora"):
        with trainer.adapter_scope_for_sample(
            adapter_type, {}, {"adapter_mode": "bypass"}, merge_size=None
        ):
            assert trainer._PLAIN_LORA_BYPASS.get() is True
        with trainer.adapter_scope_for_sample(
            adapter_type, {}, {"adapter_mode": "active"}, merge_size=None
        ):
            assert trainer._PLAIN_LORA_BYPASS.get() is False
        # P1-4 的 per-negative 冻结锚点:显式覆盖优先于样本字段
        with trainer.adapter_scope_for_sample(
            adapter_type,
            {},
            {"adapter_mode": "active"},
            merge_size=None,
            adapter_mode="bypass",
        ):
            assert trainer._PLAIN_LORA_BYPASS.get() is True
    with pytest.raises(ValueError, match="unknown adapter_type"):
        trainer.adapter_scope_for_sample(
            "made_up", {}, {"adapter_mode": "bypass"}, merge_size=None
        )
    with pytest.raises(ValueError, match="unknown adapter_mode"):
        trainer.adapter_scope_for_sample(
            "full_policy_lora", {}, {"adapter_mode": "mystery"}, merge_size=None
        )


def test_active_scope_pins_bypass_off_even_when_nested() -> None:
    # 外层 bypass 不得泄漏进 active 前向 —— active 是显式 False,不是"不设置"。
    with trainer.plain_lora_bypass_scope(True):
        with trainer.adapter_scope_for_sample(
            "ungated_kv_lora", {}, {"adapter_mode": "active"}, merge_size=None
        ):
            assert trainer._PLAIN_LORA_BYPASS.get() is False
        assert trainer._PLAIN_LORA_BYPASS.get() is True


def test_scorer_arm_partition_matches_the_desktop_contract() -> None:
    import scripts.score_sparse_history_arms as scorer

    assert scorer.arm_partition_for_schema(trainer.SPARSE_DESKTOP_SAMPLE_SCHEMA) == (
        ("R0", "S0"),
        ("RA", "SA"),
    )
    assert scorer.arm_partition_for_schema(trainer.SPARSE_SAMPLE_SCHEMA) == (
        scorer.FROZEN_ARM_SLOTS,
        scorer.ACTIVE_ARM_SLOTS,
    )


def test_epoch_tail_plan_fixes_only_mixed_partitions() -> None:
    """混合尾巴(v3 死锁案发现场)补同步;均匀场景保持旧行为逐字节不变。"""
    plan = trainer.sparse_epoch_tail_plan
    # v3 桌面语料:1,774 单元 → stride 分片 444/444/443/443,混合 → 死锁案
    lengths = [len(list(range(1774))[r::4]) for r in range(4)]
    assert lengths == [444, 444, 443, 443]
    assert plan(lengths, 4) == [False, False, True, True]
    # v1(773)与 v5 mobile(6,807):全带尾巴 → 全 False(旧行为)
    assert plan([len(list(range(773))[r::4]) for r in range(4)], 4) == [False] * 4
    assert plan([len(list(range(6807))[r::4]) for r in range(4)], 4) == [False] * 4
    # 全整除 → 全 False;rank0 独带尾 → 只补 rank0
    assert plan([444, 444, 444, 444], 4) == [False] * 4
    assert plan([445, 444, 444, 444], 4) == [True, False, False, False]


def test_sparse_supported_adapter_types_are_the_ablation_rows() -> None:
    assert trainer.SPARSE_SUPPORTED_ADAPTER_TYPES == (
        "history_gated_kv",
        "full_policy_lora",
        "ungated_kv_lora",
    )


def test_did_loss_reports_sa_minus_wa_on_a_desktop_group(tmp_path) -> None:
    samples, group = as_group(corpus_rows(tmp_path, split="train"))
    recorder = _Recorder(
        {"SA": 0.10, "S0": 0.05, "RA": 0.02, "R0": 0.0, "WA": -0.30, "WA|bypass": -0.32}
    )
    diagnostics: dict[str, float] = {}
    trainer._sparse_history_group_loss(
        samples=samples,
        group=group,
        forward=recorder,
        training={},
        adapter_parameters=[],
        accumulation=1,
        torch=None,
        objective_kind=trainer.SPARSE_OBJECTIVE_DID_RA_AWARE,
        diagnostics_out=diagnostics,
    )
    assert diagnostics["negatives"] == 1.0
    assert diagnostics["SA_minus_WA"] == pytest.approx(0.10 - (-0.30))
    assert diagnostics["wrong_drift_abs"] == pytest.approx(0.02)
    assert diagnostics["did_content_wrong"] == pytest.approx(
        (0.10 - 0.05) - ((-0.30) - (-0.32))
    )
    # W0 = 同一行 WA 的临时 bypass 前向(交接 §6 的 W0 定义)
    assert ("WA", False, "bypass") in recorder.calls
