#!/usr/bin/env python3
"""Sparse-history 目标函数与 adapter 开关的契约测试(pytest,默认无需 torch/GPU)。

# note (luojiaxuan): 第二轮重写。第一轮这套用例是与实现**并行**写的,断言对着一份
# 想象中的 API,实跑 4 PASS / 26 FAIL —— 一份跑不起来的测试比没有测试更危险,因为
# 它在审计材料里长得像"已验证"。本轮的每一条断言都对着 scripts.train_success_sft_lora
# 里**真实存在**的符号与签名(用 inspect.signature 核过),并且真的执行过。
#
# 相对第一轮修掉的四处结构性错配:
#   1. Recorder.__call__ 现在与 sparse_history_group_unit_loss 内层 forward 闭包的
#      真实签名一致 —— ``forward(slot, *, grad, backward_weight=None,
#      adapter_mode=None)``。少了 adapter_mode,P1-4 的 per-negative 冻结锚点前向
#      会让每一条损失用例 TypeError;
#   2. anchor 语义按**修好后的定义**重写:负样本锚到**它自己的 bypass 分数**,而不
#      是锚到 R0。第一轮那条 test_anchor_pulls_a_raised_negative_back_to_the_reference
#      断言的是旧语义(锚到参考臂),与正确修复直接矛盾,整条删除并由
#      test_drift_anchors_each_negative_to_its_own_frozen_score 取代;
#   3. 损失返回类型统一为 ``float | None``,诊断只走 ``diagnostics_out`` 原地填充;
#   4. build_sparse_history_units(samples) 只吃一个位置参数,没有 training= kwarg;
#      split 的唯一权威是样本的 ``split`` 字段。
#
# 另外:负样本份额按**组内实际存在**的负样本归一化(scale / Σscale),所以"duplicate
# 单独成组时份额是 1.0、和别的负样本共存时才是 0.5/Σ"是契约的一部分,不是 bug。
# 第一轮那条 test_duplicate_negative_is_half_weighted 按未归一化算,数字本身就是错的。
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import sys
import types
from pathlib import Path
from typing import Any, Iterator

import pytest


# ---------------------------------------------------------------------------
# 仓库路径与重依赖桩(在 import 被测模块之前完成)
# ---------------------------------------------------------------------------
def _repo_code_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "causalcache").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository code root from the test file")


_CODE_ROOT = _repo_code_root()
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

STUBBED: set[str] = set()


def stub_if_missing(name: str, **attributes: Any) -> None:
    """Register a stand-in module only when the real dependency is unavailable.

    # note (luojiaxuan): 只桩掉装不上的重依赖(torch / PIL / 推理 runtime),被测模块
    # 本身永远走真实 import。哪些被桩了记录在 ``STUBBED``,需要真 torch 的用例据此
    # skip,避免拿桩当真实现骗过断言。
    """
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - 任何 import 失败都退回桩
        pass
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    STUBBED.add(name)
    parent_name, _, leaf = name.rpartition(".")
    if parent_name:
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, leaf, module)


stub_if_missing("torch")
stub_if_missing("PIL")
stub_if_missing("PIL.Image", open=lambda *args, **kwargs: None)
stub_if_missing(
    "causalcache.policy.gui_owl_v2_1_runtime", GUIOwlV21OfficialToolsRuntime=object
)
stub_if_missing(
    "scripts.run_exploratory_closed_loop_episode",
    EFFECTIVE_VISUAL_TOKENS_PER_IMAGE=2560,
)
stub_if_missing(
    "causalcache.policy.history_adapter_context",
    HistoryAdapterContext=object,
    history_adapter_scope=lambda context: contextlib.nullcontext(),
    get_history_adapter_context=lambda: None,
)

from scripts import train_success_sft_lora as trainer  # noqa: E402

TORCH_IS_STUBBED = "torch" in STUBBED
requires_real_torch = pytest.mark.skipif(
    TORCH_IS_STUBBED, reason="需要真实 torch(纯逻辑部分用桩已覆盖)"
)


# ---------------------------------------------------------------------------
# 五臂契约与样本工厂
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "causalcache.sparse_history_sample.v2"
ARM_SPEC: dict[str, tuple[str, str, str, str, str]] = {
    "N0": ("N0", "deployment_baseline", "official_multiturn", "recent", "bypass"),
    "R0": ("R0", "reference", "sparse_single_turn", "recent", "bypass"),
    "S0": ("S0", "measurement", "sparse_single_turn", "sparse", "bypass"),
    "RA": ("RA", "measurement", "sparse_single_turn", "recent", "active"),
    "SA": ("SA", "positive", "sparse_single_turn", "sparse", "active"),
    "SA_neg_step_shuffled": (
        "SA", "negative", "sparse_single_turn", "sparse", "active",
    ),
    "SA_neg_irrelevant": ("SA", "negative", "sparse_single_turn", "sparse", "active"),
    "SA_neg_duplicate": ("SA", "negative", "sparse_single_turn", "sparse", "active"),
}
NEGATIVE_SLOTS = (
    "SA_neg_step_shuffled", "SA_neg_irrelevant", "SA_neg_duplicate",
)
DEFAULT_NEGATIVE_SCALE = {
    "SA_neg_step_shuffled": 1.0,
    "SA_neg_irrelevant": 1.0,
    "SA_neg_duplicate": 0.5,
}
NEGATIVE_KIND = {
    "SA_neg_step_shuffled": "step_shuffled",
    "SA_neg_irrelevant": "irrelevant",
    "SA_neg_duplicate": "duplicate",
}
# 与 _sparse_history_group_loss 的 training.get(...) 缺省值逐字一致
GAIN_MARGIN = 0.01
RANK_MARGIN = 0.02
GAIN_WEIGHT = 1.0
RANK_WEIGHT = 1.0
DRIFT_WEIGHT = 2.0
TRAINING = {"sparse_history": True, "history_lora_l2_weight": 0.0}


def _messages_with_images(budget: int, images: list[str], current: str) -> list[dict]:
    """Minimal message list carrying exactly ``budget + 1`` image parts.

    # note (luojiaxuan): validate_sparse_sample 会数 messages 里的 image part 并要求
    # 恰好等于 budget+1,第一轮的工厂只放了一个 text part,于是任何走校验的用例
    # (单元构造 / 留出集)都在 fail-closed 上翻车。图像 part 只带 path,与落盘样本一致。
    """
    parts: list[dict[str, Any]] = [{"type": "image", "path": path} for path in images]
    parts.append({"type": "text", "text": "What is the next step?"})
    parts.append({"type": "image", "path": current})
    assert sum(1 for part in parts if part["type"] == "image") == budget + 1
    return [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        {"role": "user", "content": parts},
    ]


def make_sample(
    slot: str,
    *,
    episode: str = "EP",
    decision_step: int = 9,
    budget: int = 2,
    split: str = "train",
    **overrides: Any,
) -> dict[str, Any]:
    arm_id, role, prompt_format, selection_mode, adapter_mode = ARM_SPEC[slot]
    steps = (
        list(range(decision_step - budget, decision_step))
        if selection_mode == "recent"
        else [2, 5, 8, 3 + budget][:budget]
    )
    steps = sorted(set(steps))[:budget]
    images = [f"images/{episode}/obs-{step - 1:03d}.png" for step in steps]
    current = f"images/{episode}/obs-{decision_step - 1:03d}.png"
    pair_group = f"{episode}:{decision_step}"
    sample: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"{pair_group}|{slot}",
        "pair_group": pair_group,
        "episode": episode,
        "decision_step": decision_step,
        "arm_slot": slot,
        "arm_id": arm_id,
        "role": role,
        "prompt_format": prompt_format,
        "selection_mode": selection_mode,
        "adapter_mode": adapter_mode,
        "budget": budget,
        "selected_steps": steps,
        "selected_images": images,
        "current_image": current,
        "target_text": "Action: go\n<tool_call>\n{}\n</tool_call>",
        "messages": _messages_with_images(budget, images, current),
        "split": split,
        "reference_arm_id": "R0",
        "deployment_baseline_arm_id": "N0",
    }
    if role == "negative":
        sample["negative_kind"] = NEGATIVE_KIND[slot]
        sample["negative_scale"] = DEFAULT_NEGATIVE_SCALE[slot]
        if sample["negative_kind"] == "irrelevant":
            sample["donor_episode"] = f"{episode}-donor"
    sample.update(overrides)
    return sample


def make_group(
    slots: tuple[str, ...] = tuple(ARM_SPEC),
    *,
    episode: str = "EP",
    split: str = "train",
    budget: int = 2,
    decision_step: int = 9,
    overrides: dict[str, dict[str, Any]] | None = None,
    extra: tuple[dict[str, Any], ...] = (),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return (samples, group) with group mapping arm_slot -> index.

    # note (luojiaxuan): per-slot overrides 会先并进工厂 kwargs 再展开,所以覆盖
    # ``budget``/``decision_step`` 不再触发 "got multiple values for keyword
    # argument 'budget'"(第一轮的纯测试端 bug),而且覆盖后的 budget 会真的驱动
    # selected_steps / selected_images / messages 的图数,不至于造出自相矛盾的样本。
    """
    overrides = overrides or {}
    samples: list[dict[str, Any]] = []
    for slot in slots:
        kwargs: dict[str, Any] = {
            "episode": episode,
            "split": split,
            "budget": budget,
            "decision_step": decision_step,
        }
        kwargs.update(overrides.get(slot, {}))
        samples.append(make_sample(slot, **kwargs))
    samples.extend(dict(item) for item in extra)
    group = {sample["arm_slot"]: index for index, sample in enumerate(samples)}
    return samples, group


class Recorder:
    """Stub forward matching the real closure's signature, logging every call.

    # note (luojiaxuan): 签名必须逐字对齐 sparse_history_group_unit_loss 内层的
    # ``forward(slot, *, grad, backward_weight=None, adapter_mode=None)``。
    # ``frozen`` 是每条负样本在 **adapter bypass** 下的分数,也就是 P1-4 之后 drift
    # 的锚点;不给就等于 active,drift 恒为 0(用于只想考察 gain/rank 的用例)。
    # 这里同样复刻真实闭包对 override 的 fail-closed 限制:override 只允许出现在
    # no-grad + bypass 的取值上,否则损失若哪天用它去拿带梯度的 active 分数,
    # 测试要当场炸而不是默默给出一个数。
    """

    def __init__(
        self,
        values: dict[str, float],
        *,
        frozen: dict[str, float] | None = None,
        shift: float = 0.0,
    ) -> None:
        self.values = {slot: value + shift for slot, value in values.items()}
        self.frozen = {
            slot: value + shift for slot, value in (frozen or {}).items()
        }
        self.calls: list[tuple[str, bool, float | None, str | None]] = []

    def __call__(
        self,
        slot: str,
        *,
        grad: bool,
        backward_weight: float | None = None,
        adapter_mode: str | None = None,
    ) -> float | None:
        if adapter_mode is not None and (grad or adapter_mode != "bypass"):
            raise ValueError(
                "adapter_mode override is reserved for the no-grad bypass anchor"
            )
        self.calls.append((slot, grad, backward_weight, adapter_mode))
        if adapter_mode == "bypass" and slot in self.frozen:
            return self.frozen[slot]
        if adapter_mode == "bypass":
            return self.values.get(slot)
        return self.values.get(slot)

    @property
    def grad_weights(self) -> dict[str, float]:
        return {
            slot: weight
            for slot, grad, weight, _mode in self.calls
            if grad and weight is not None
        }

    def touched(self, slot: str) -> bool:
        return any(call[0] == slot for call in self.calls)

    def bypass_anchor_calls(self) -> list[str]:
        return [slot for slot, _g, _w, mode in self.calls if mode == "bypass"]


LOSS_SIGNATURE = inspect.signature(trainer._sparse_history_group_loss)


def call_loss(
    samples: list[dict[str, Any]],
    group: dict[str, int],
    forward: Any,
    *,
    training: dict[str, Any] | None = None,
    accumulation: int = 1,
    adapter_parameters: list[Any] | None = None,
    diagnostics_out: dict[str, float] | None = None,
) -> float | None:
    """Invoke the group loss through its real signature (no **kwargs shortcuts)."""
    # 每次调用清空滚动打印窗口:它是进程级列表,累积到 25 的倍数就往 stdout 打 JSON,
    # 与被测语义无关,却会把测试输出淹掉。
    trainer._SPARSE_DIAG.clear()
    return trainer._sparse_history_group_loss(
        samples=samples,
        group=group,
        forward=forward,
        training=dict(TRAINING if training is None else training),
        adapter_parameters=list(adapter_parameters or []),
        accumulation=accumulation,
        torch=None,
        diagnostics_out=diagnostics_out,
    )


# ---------------------------------------------------------------------------
# 1. 导入与签名:测试对着仓库里真实存在的 API
# ---------------------------------------------------------------------------
def test_trainer_imports_from_repository() -> None:
    module = importlib.import_module("scripts.train_success_sft_lora")
    assert Path(module.__file__).resolve().is_relative_to(_CODE_ROOT)
    assert callable(module._sparse_history_group_loss)


def test_loss_signature_is_the_one_the_tests_call() -> None:
    """签名断言前置:签名一变,失败点是这一条而不是 20 条语义用例一起爆。"""
    parameters = LOSS_SIGNATURE.parameters
    assert set(parameters) == {
        "samples", "group", "forward", "training", "adapter_parameters",
        "accumulation", "torch", "diagnostics_out",
    }
    assert "wrapped" not in parameters, "LoRA 参数入参叫 adapter_parameters"
    assert parameters["diagnostics_out"].default is None


def test_forward_closure_signature_matches_the_recorder_stub() -> None:
    """Recorder 必须与真实 forward 闭包同签名,否则这套桩测的是想象中的 API。

    # note (luojiaxuan): 第一轮全盘失败的根因就是这里 —— 闭包早已带 adapter_mode
    # (P1-4 的 per-negative 冻结锚点),桩却没有。闭包 import 不出来,只能从
    # sparse_history_group_unit_loss 的源码里把它的 def 抠出来比对。
    """
    source = inspect.getsource(trainer.sparse_history_group_unit_loss)
    for fragment in ("def forward(", "slot: str,", "grad: bool", "backward_weight",
                     "adapter_mode: str | None = None"):
        assert fragment in source, f"forward 闭包签名缺少 {fragment!r}"
    recorder = inspect.signature(Recorder.__call__).parameters
    assert set(recorder) == {"self", "slot", "grad", "backward_weight", "adapter_mode"}


def test_trainer_has_no_name_prefix_control_flow() -> None:
    """审计 P0-4:adapter/参考臂/负样本都不得靠 variant 名字前缀判断。"""
    source = inspect.getsource(trainer)
    for forbidden in (
        'variant.startswith("native_recent")',
        'variant.startswith("sameformat_recent")',
        "SPARSE_NEGATIVE_SCALE",
        "sameformat_recent",
    ):
        assert forbidden not in source, f"仍有按名字前缀的控制流:{forbidden}"


# ---------------------------------------------------------------------------
# 2. 返回类型与诊断通道
# ---------------------------------------------------------------------------
def test_loss_returns_float_or_none_never_a_dict() -> None:
    samples, group = make_group()
    satisfied = call_loss(samples, group, Recorder({**{s: 0.0 for s in group}, "SA": 1.0}))
    assert satisfied is None, "全部达标且无 drift 的组按跳过处理"
    violating = call_loss(samples, group, Recorder({**{s: 0.0 for s in group}, "SA": -0.05}))
    assert isinstance(violating, float) and not isinstance(violating, dict)


def test_diagnostics_are_delivered_through_diagnostics_out_only() -> None:
    samples, group = make_group()
    values = {slot: 0.0 for slot in group}
    values["SA"] = -0.05
    frozen = {"SA_neg_step_shuffled": -0.03}
    diagnostics: dict[str, float] = {}
    result = call_loss(
        samples, group, Recorder(values, frozen=frozen), diagnostics_out=diagnostics
    )
    assert isinstance(result, float)
    assert diagnostics["loss"] == pytest.approx(result, abs=1e-12)
    assert diagnostics["negatives"] == 3.0
    assert diagnostics["SA_minus_R0"] == pytest.approx(-0.05, abs=1e-12)
    assert diagnostics["step_shuffled_drift_abs"] == pytest.approx(0.03, abs=1e-12)


def test_diagnostic_keys_are_spelled_like_the_gate_vocabulary() -> None:
    """审计第 10 条:损失诊断的键必须与 config.gates 的量名逐字相同。"""
    samples, group = make_group()
    values = {slot: 0.0 for slot in group}
    values["SA"] = -0.05
    diagnostics: dict[str, float] = {}
    call_loss(samples, group, Recorder(values), diagnostics_out=diagnostics)
    runtime_only = {"loss", "negatives"}
    quantities = set(diagnostics) - runtime_only
    assert quantities <= set(trainer.SPARSE_GATE_VOCABULARY), (
        f"诊断键 {sorted(quantities - set(trainer.SPARSE_GATE_VOCABULARY))} 不在 gate 词表里"
    )
    for kind in trainer.SPARSE_NEGATIVE_KINDS:
        gap_key, drift_key = trainer.sparse_diagnostic_keys(kind)
        assert gap_key in diagnostics and drift_key in diagnostics
        assert gap_key in trainer.SPARSE_REQUIRED_GATES
        assert drift_key in trainer.SPARSE_REQUIRED_GATES


# ---------------------------------------------------------------------------
# 3. 损失的数值契约
# ---------------------------------------------------------------------------
def test_satisfied_group_is_skipped_without_backward() -> None:
    samples, group = make_group()
    values = {slot: 0.0 for slot in group}
    values["SA"] = 1.0
    forward = Recorder(values)
    assert call_loss(samples, group, forward) is None
    assert forward.grad_weights == {}


def test_gain_and_three_rank_hinges_add_up() -> None:
    """三个负样本的份额之和恒为 1:总 rank 质量与 K、与负样本个数无关。"""
    samples, group = make_group()
    values = {slot: 0.0 for slot in group}
    values["SA"] = -0.05
    forward = Recorder(values)
    result = call_loss(samples, group, forward)
    expected = GAIN_WEIGHT * (GAIN_MARGIN + 0.05) + RANK_WEIGHT * (RANK_MARGIN + 0.05)
    assert result == pytest.approx(expected, abs=1e-9)
    assert forward.grad_weights["SA"] < 0.0, "ℓc 落后时梯度必须推高 ℓc"


def test_negative_shares_are_normalised_over_the_present_negatives() -> None:
    """duplicate 单独成组时份额是 1.0,不是它的原始 scale 0.5。

    # note (luojiaxuan): 归一化的分母只能是**本组真正进入损失**的负样本。按固定
    # scale 求和会让 K=1 组(只剩 irrelevant)的负样本质量只有别组的 40%,等于按
    # 预算给梯度加权;按 NEGATIVE_SPECS 的全集归一化则更糟(份额 0.2)。这条用例
    # 同时把这两种写法钉死。
    """
    samples, group = make_group(slots=("R0", "SA", "SA_neg_duplicate"))
    forward = Recorder({"R0": 0.0, "SA": 0.0, "SA_neg_duplicate": 0.0})
    result = call_loss(samples, group, forward)
    assert result == pytest.approx(GAIN_MARGIN + 1.0 * RANK_MARGIN, abs=1e-9)


def test_k1_inventory_keeps_the_full_negative_mass() -> None:
    """K=1 组只剩 irrelevant(shuffled/duplicate 与 SA 逐字相同,不入库)。"""
    samples, group = make_group(
        slots=("N0", "R0", "S0", "RA", "SA", "SA_neg_irrelevant"), budget=1
    )
    forward = Recorder({slot: 0.0 for slot in group})
    result = call_loss(samples, group, forward)
    assert result == pytest.approx(GAIN_MARGIN + 1.0 * RANK_MARGIN, abs=1e-9)
    assert forward.grad_weights["SA_neg_irrelevant"] == pytest.approx(1.0, abs=1e-12)


def test_negative_scale_is_read_from_the_field_not_a_name_table() -> None:
    """把 duplicate 的 scale 改成 0.25,份额必须跟着变;不变说明在查名字硬编码表。"""
    slots = ("R0", "SA", "SA_neg_duplicate", "SA_neg_irrelevant")
    values = {"R0": 0.0, "SA": 0.0, "SA_neg_duplicate": -0.10, "SA_neg_irrelevant": 0.0}

    default_samples, default_group = make_group(slots=slots)
    default_result = call_loss(default_samples, default_group, Recorder(values))
    # scale_mass = 0.5 + 1.0;duplicate 的 rank hinge 未触发(0.02 - 0.10 < 0)
    assert default_result == pytest.approx(
        GAIN_MARGIN + (1.0 / 1.5) * RANK_MARGIN, abs=1e-9
    )

    retuned_samples, retuned_group = make_group(
        slots=slots, overrides={"SA_neg_duplicate": {"negative_scale": 0.25}}
    )
    retuned_result = call_loss(retuned_samples, retuned_group, Recorder(values))
    assert retuned_result == pytest.approx(
        GAIN_MARGIN + (1.0 / 1.25) * RANK_MARGIN, abs=1e-9
    )
    assert retuned_result != pytest.approx(default_result, abs=1e-9)


def test_negative_membership_comes_from_role_not_slot_name() -> None:
    """诊断字段 variant 写成负样本名、role 却是 measurement 的臂不得进 rank/drift。"""
    decoy = make_sample("S0", variant="sparse_step_shuffled", negative_scale=1.0)
    samples, group = make_group(slots=("R0", "SA"), extra=(decoy,))
    forward = Recorder({"R0": 0.0, "SA": 0.0, "S0": -5.0})
    result = call_loss(samples, group, forward)
    assert result == pytest.approx(GAIN_MARGIN, abs=1e-9)
    assert not forward.touched("S0"), "S0 是测量臂,不进训练损失"


def test_measurement_arms_stay_out_of_the_training_loss() -> None:
    samples, group = make_group()
    values = {slot: 0.0 for slot in group}
    values["SA"] = -0.05
    forward = Recorder(values)
    call_loss(samples, group, forward)
    for slot in ("N0", "S0", "RA"):
        assert not forward.touched(slot), f"{slot} 只在留出集打分,不进训练损失"


@pytest.mark.parametrize("shift", (-3.0, 0.0, 2.5))
def test_loss_is_invariant_to_a_global_logprob_shift(shift: float) -> None:
    """只优化条件之间的相对 log-prob:整体平移不改变损失。含 CE 项就会破。"""
    samples, group = make_group()
    values = {slot: 0.0 for slot in group}
    values["SA"] = -0.05
    values["SA_neg_irrelevant"] = 0.2
    frozen = {"SA_neg_irrelevant": 0.1, "SA_neg_duplicate": -0.05}
    baseline = call_loss(samples, group, Recorder(values, frozen=frozen))
    shifted = call_loss(samples, group, Recorder(values, frozen=frozen, shift=shift))
    assert baseline is not None
    assert shifted == pytest.approx(baseline, abs=1e-9)


def test_no_cross_entropy_term_in_the_sparse_objective() -> None:
    source = inspect.getsource(trainer._sparse_history_group_loss)
    assert "ce_weight" not in source, "sparse 目标的 CE 权重恒为 0,不得再读 ce 权重"


# ---------------------------------------------------------------------------
# 4. drift 锚点:锚到**自身冻结分数**,不再锚到 R0(P1-4 的新语义)
# ---------------------------------------------------------------------------
def test_drift_anchors_each_negative_to_its_own_frozen_score() -> None:
    """负样本停在自己的冻结分数上就不该受罚,哪怕它离 R0 很远。

    # note (luojiaxuan): 这条取代第一轮的
    # test_anchor_pulls_a_raised_negative_back_to_the_reference —— 那条按旧语义
    # (SmoothL1(ℓn_active, ℓr_frozen))断言,等价于假设 shuffled/irrelevant/duplicate
    # 在冻结模型上本就该等于 recent-K 参考臂。这条假设不成立,旧 anchor 会把 adapter
    # 往一个错误的常数上拽并与 rank 项对冲。新语义只惩罚 **adapter 造成的位移**。
    """
    samples, group = make_group(slots=("R0", "SA", "SA_neg_irrelevant"))
    # 冻结模型本来就偏好这条负样本(0.3 ≫ R0 的 0.0),但 adapter 没有动它
    forward = Recorder(
        {"R0": 0.0, "SA": 0.5, "SA_neg_irrelevant": 0.3},
        frozen={"SA_neg_irrelevant": 0.3},
    )
    assert call_loss(samples, group, forward) is None, (
        "ℓn 停在自身冻结分数上时 drift 为 0;旧的锚到 R0 语义会在这里收 2*0.5*0.3² 的罚"
    )
    assert forward.grad_weights == {}


def test_drift_penalises_only_the_adapter_induced_displacement() -> None:
    samples, group = make_group(slots=("R0", "SA", "SA_neg_irrelevant"))
    forward = Recorder(
        {"R0": 0.0, "SA": 0.5, "SA_neg_irrelevant": 0.3},
        frozen={"SA_neg_irrelevant": 0.0},
    )
    result = call_loss(samples, group, forward)
    # rank hinge 未触发(0.5-0.3=0.2 > 0.02);只剩 drift = 2.0 * 1.0 * 0.5 * 0.3²
    assert result == pytest.approx(DRIFT_WEIGHT * 1.0 * 0.5 * 0.3 * 0.3, abs=1e-9)
    assert forward.grad_weights["SA_neg_irrelevant"] == pytest.approx(
        DRIFT_WEIGHT * 1.0 * 0.3, abs=1e-12
    ), "梯度必须把被 adapter 抬高的负样本压回它自己的冻结分数"


def test_drift_is_independent_of_where_the_reference_sits() -> None:
    """把 R0 整体挪走,drift 项一分不变 —— 旧的锚到 R0 语义在这里必然改变。"""
    samples, group = make_group(slots=("R0", "SA", "SA_neg_irrelevant"))
    frozen = {"SA_neg_irrelevant": 0.0}
    near = call_loss(
        samples, group,
        Recorder({"R0": 0.0, "SA": 0.5, "SA_neg_irrelevant": 0.3}, frozen=frozen),
    )
    far = call_loss(
        samples, group,
        Recorder({"R0": -0.4, "SA": 0.5, "SA_neg_irrelevant": 0.3}, frozen=frozen),
    )
    assert near is not None and far == pytest.approx(near, abs=1e-9)


def test_frozen_anchor_is_a_no_grad_bypass_forward_of_the_same_negative() -> None:
    """每条负样本恰好多一次 no-grad + adapter_mode='bypass' 前向;正样本没有。"""
    samples, group = make_group()
    values = {slot: 0.0 for slot in group}
    values["SA"] = -0.05
    forward = Recorder(values)
    call_loss(samples, group, forward)
    assert sorted(forward.bypass_anchor_calls()) == sorted(NEGATIVE_SLOTS)
    for slot, grad, _weight, mode in forward.calls:
        if mode is not None:
            assert mode == "bypass" and not grad, (
                "adapter_mode 覆盖只准用于 no-grad 的 bypass 锚点"
            )
        if slot in ("SA", "R0"):
            assert mode is None, f"{slot} 的分数由样本自己的 adapter_mode 决定"


def test_a_negative_whose_anchor_forward_fails_leaves_the_normaliser() -> None:
    """取不到冻结锚点的负样本整条退出损失,剩下两条的份额之和仍是 1。

    # note (luojiaxuan): 旧写法先按全部负样本归一化、再在循环里 continue 掉取不到值
    # 的项,剩下份额之和 < 1,等于按"哪些前向恰好失败"给该组梯度打折扣。
    """
    samples, group = make_group()
    values = {slot: 0.0 for slot in group}
    values["SA"] = -0.05

    class Dropping(Recorder):
        def __call__(self, slot: str, **kwargs: Any) -> float | None:
            if slot == "SA_neg_duplicate" and kwargs.get("adapter_mode") == "bypass":
                super().__call__(slot, **kwargs)
                return None
            return super().__call__(slot, **kwargs)

    forward = Dropping(values)
    diagnostics: dict[str, float] = {}
    result = call_loss(samples, group, forward, diagnostics_out=diagnostics)
    assert diagnostics["negatives"] == 2.0
    assert "duplicate_drift_abs" not in diagnostics
    expected = GAIN_WEIGHT * (GAIN_MARGIN + 0.05) + RANK_WEIGHT * (RANK_MARGIN + 0.05)
    assert result == pytest.approx(expected, abs=1e-9)
    assert "SA_neg_duplicate" not in forward.grad_weights


# ---------------------------------------------------------------------------
# 5. reference 的选择:读 reference_arm_id 字段,不猜前缀
# ---------------------------------------------------------------------------
def test_reference_is_taken_from_the_reference_arm_id_field() -> None:
    decoy = make_sample("S0", variant="native_recent2")
    decoy["arm_slot"] = "native_recent2"
    samples, group = make_group(slots=("R0", "SA"), extra=(decoy,))
    forward = Recorder({"R0": 0.0, "SA": -0.05, "native_recent2": 10.0})
    result = call_loss(samples, group, forward)
    assert forward.touched("R0")
    assert not forward.touched("native_recent2"), (
        "参考臂必须来自 reference_arm_id=R0,不得因为名字像 native_recent 就被选中"
    )
    assert result == pytest.approx(GAIN_MARGIN + 0.05, abs=1e-9)


def test_reference_forward_never_carries_gradient() -> None:
    samples, group = make_group()
    values = {slot: 0.0 for slot in group}
    values["SA"] = -0.05
    forward = Recorder(values)
    call_loss(samples, group, forward)
    assert all(
        not grad for slot, grad, _weight, _mode in forward.calls if slot == "R0"
    ), "ℓr 是冻结参考,只能 no-grad 取值"
    assert "R0" not in forward.grad_weights


def test_inconsistent_reference_arm_id_is_rejected() -> None:
    samples, group = make_group(
        overrides={"SA_neg_irrelevant": {"reference_arm_id": "N0"}}
    )
    forward = Recorder({slot: 0.0 for slot in group})
    with pytest.raises(ValueError):
        call_loss(samples, group, forward)


@pytest.mark.parametrize("dropped", ("R0", "SA"))
def test_missing_required_arm_is_fail_closed(dropped: str) -> None:
    """审计第 8 条:缺 R0/SA 必须抛错,不得 return None 让分母悄悄变小。"""
    slots = tuple(slot for slot in ARM_SPEC if slot != dropped)
    samples, group = make_group(slots=slots)
    forward = Recorder({slot: 0.0 for slot in group})
    with pytest.raises((ValueError, KeyError)):
        call_loss(samples, group, forward)


def test_reference_must_be_budget_and_format_matched() -> None:
    """预算/格式不匹配由 validate_sparse_group 拦下 —— 这是组校验的职责。"""
    samples, group = make_group(overrides={"R0": {"budget": 4}})
    with pytest.raises(ValueError):
        trainer.validate_sparse_group(samples, pair_group="EP:9", group=group)
    format_samples, format_group = make_group(
        overrides={"R0": {"prompt_format": "official_multiturn"}}
    )
    with pytest.raises(ValueError):
        trainer.validate_sparse_group(
            format_samples, pair_group="EP:9", group=format_group
        )


def test_a_healthy_group_passes_validate_sparse_group() -> None:
    samples, group = make_group()
    assert trainer.validate_sparse_group(samples, pair_group="EP:9", group=group) == "R0"


# ---------------------------------------------------------------------------
# 6. adapter bypass / active 语义(唯一入口 adapter_context_for_sample)
# ---------------------------------------------------------------------------
class _FakeCount:
    def __init__(self, value: int) -> None:
        self.value = value

    def sum(self) -> int:
        return self.value


class _FakeLabels:
    """Minimal stand-in supporting ``int((labels != -100).sum())``."""

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def __ne__(self, other: Any) -> Any:  # noqa: D105
        return _FakeCount(self.target_length)

    __hash__ = None  # type: ignore[assignment]


class _FakeIds:
    def __init__(self, length: int) -> None:
        self.shape = (1, length)


class _FakeContext:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


def fake_encoded(*, length: int = 13) -> dict[str, Any]:
    return {
        "input_ids": _FakeIds(length),
        "labels": _FakeLabels(2),
        "mm_token_type_ids": object(),
        "image_grid_thw": object(),
    }


@contextlib.contextmanager
def stubbed_adapter_modules() -> Iterator[None]:
    """Swap the mask/context modules so the dispatch runs without torch.

    # note (luojiaxuan): 用上下文管理器而不是 pytest fixture,是因为 fixture 的
    # 恢复语义依赖 monkeypatch.setitem 对"键本来不存在"的处理;这里手动 save/restore
    # 到底,免得把 None 留在 sys.modules 里毒化后面的 import。
    """
    names = (
        "causalcache.policy.history_token_roles",
        "causalcache.policy.history_adapter_context",
    )
    saved = {name: sys.modules.get(name, ...) for name in names}
    roles = types.ModuleType(names[0])
    roles.build_history_token_mask = (  # type: ignore[attr-defined]
        lambda input_ids, mm, grid, count, merge_size: _FakeCount(4 * count)
    )
    roles.assert_mask_disjoint = lambda mask, target_start: None  # type: ignore[attr-defined]
    context_module = types.ModuleType(names[1])
    context_module.HistoryAdapterContext = _FakeContext  # type: ignore[attr-defined]
    context_module.history_adapter_scope = (  # type: ignore[attr-defined]
        lambda ctx: contextlib.nullcontext()
    )
    context_module.get_history_adapter_context = lambda: None  # type: ignore[attr-defined]
    sys.modules[names[0]] = roles
    sys.modules[names[1]] = context_module
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is ...:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_adapter_dispatch_is_exported_with_the_expected_signature() -> None:
    dispatch = trainer.adapter_context_for_sample
    parameters = inspect.signature(dispatch).parameters
    assert list(parameters) == ["encoded", "sample", "merge_size", "adapter_mode", "slot"]
    assert parameters["adapter_mode"].default is None
    assert parameters["slot"].default is None


@pytest.mark.parametrize("slot", ("N0", "R0", "S0"))
def test_bypass_arms_get_no_adapter_context(slot: str) -> None:
    with stubbed_adapter_modules():
        assert trainer.adapter_context_for_sample(
            fake_encoded(), make_sample(slot), merge_size=2
        ) is None


@pytest.mark.parametrize("slot", ("RA", "SA", "SA_neg_irrelevant"))
def test_active_arms_get_a_non_empty_history_mask(slot: str) -> None:
    with stubbed_adapter_modules():
        context = trainer.adapter_context_for_sample(
            fake_encoded(), make_sample(slot), merge_size=2
        )
    assert context is not None
    assert int(context.history_token_mask.sum()) > 0


def test_explicit_bypass_override_beats_an_active_sample_field() -> None:
    """P1-4 的冻结锚点:同一条 active 负样本被显式要求 bypass 时必须拿到 None。"""
    with stubbed_adapter_modules():
        assert trainer.adapter_context_for_sample(
            fake_encoded(),
            make_sample("SA_neg_irrelevant"),
            merge_size=2,
            adapter_mode="bypass",
            slot="SA_neg_irrelevant",
        ) is None


@pytest.mark.parametrize("mode", ("maybe", "", "Active", "active ", None))
def test_unknown_adapter_mode_is_fail_closed(mode: Any) -> None:
    sample = make_sample("SA")
    if mode is None:
        sample.pop("adapter_mode")
        expected: Any = (ValueError, KeyError)
    else:
        sample["adapter_mode"] = mode
        expected = ValueError
    with stubbed_adapter_modules():
        with pytest.raises(expected):
            trainer.adapter_context_for_sample(fake_encoded(), sample, merge_size=2)


def test_active_arm_with_an_empty_history_mask_is_rejected() -> None:
    """K=0 退化:active 臂拿不到任何历史 token,必须报错而不是静默走 bypass。"""
    sample = make_sample("SA")
    sample["memory_config"] = {"restored_event_step_ids": []}
    with stubbed_adapter_modules():
        # 几何函数对 K=0 返回 None(天然 bypass)……
        assert trainer.history_sample_context(
            fake_encoded(), sample, merge_size=2
        ) is None
        # ……但样本自称 active,策略函数必须拒绝
        with pytest.raises(ValueError):
            trainer.adapter_context_for_sample(fake_encoded(), sample, merge_size=2)


def test_a_sample_without_an_authoritative_k_is_rejected() -> None:
    """既无 memory_config 又非 sparse v2 schema:K 没有权威来源,拒绝前向。"""
    sample = make_sample("SA")
    sample["schema_version"] = "causalcache.sparse_history_sample.v1"
    with stubbed_adapter_modules():
        with pytest.raises(ValueError):
            trainer.adapter_context_for_sample(fake_encoded(), sample, merge_size=2)


def test_the_geometry_function_does_not_read_adapter_mode() -> None:
    """职责分界:history_sample_context 只算几何,分派留给 adapter_context_for_sample。"""
    source = inspect.getsource(trainer.history_sample_context)
    assert "adapter_mode" not in source
    with stubbed_adapter_modules():
        # bypass 臂在几何函数里照样能算出 mask —— 是策略函数决定不用它
        context = trainer.history_sample_context(
            fake_encoded(), make_sample("R0"), merge_size=2
        )
    assert context is not None and int(context.history_token_mask.sum()) > 0


def test_the_gate_scorer_reuses_the_trainer_dispatch() -> None:
    """训练与验收必须共用一份 adapter_mode 分派,否则五臂契约会悄悄分叉。"""
    try:
        scorer = importlib.import_module("scripts.score_sparse_history_arms")
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"打分脚本的依赖不可用:{error}")
    assert scorer.adapter_context_for_sample is trainer.adapter_context_for_sample


@requires_real_torch
def test_active_mask_covers_exactly_the_history_image_tokens() -> None:
    """真 torch 下走真实 mask 几何:K 张历史图的 token 命中,当前图不碰。"""
    import torch

    from causalcache.policy.history_token_roles import build_history_token_mask

    # [text, img#1 x4, text, current-img x4, text, target x2]
    types_row = [0] + [1] * 4 + [0] + [1] * 4 + [0, 0, 0]
    length = len(types_row)
    encoded = {
        "input_ids": torch.zeros((1, length), dtype=torch.long),
        "mm_token_type_ids": torch.tensor([types_row], dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long),
        "labels": torch.tensor([[-100] * (length - 2) + [7, 8]], dtype=torch.long),
    }
    mask = build_history_token_mask(
        encoded["input_ids"],
        encoded["mm_token_type_ids"],
        encoded["image_grid_thw"],
        1,
        1,
    )
    assert int(mask.sum()) == 4
    assert bool(mask[0, 1:5].all())
    assert not bool(mask[0, 6:10].any()), "当前观测图的 token 不得进历史 mask"

    active = trainer.adapter_context_for_sample(
        encoded, make_sample("SA", budget=1), merge_size=1
    )
    assert active is not None and int(active.history_token_mask.sum()) == 4
    assert trainer.adapter_context_for_sample(
        encoded, make_sample("R0", budget=1), merge_size=1
    ) is None


# ---------------------------------------------------------------------------
# 7. split 与分组:读字段,不重算 hash
# ---------------------------------------------------------------------------
def test_unit_builder_signature_takes_only_samples() -> None:
    parameters = inspect.signature(trainer.build_sparse_history_units).parameters
    assert list(parameters) == ["samples"], (
        "split 的权威只有样本的 split 字段;单元构造不再接受 training= 之类的旁路"
    )


def test_split_comes_from_the_sample_field(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("trainer 不得重算 heldout hash;split 以样本字段为准")

    monkeypatch.setattr(trainer, "heldout_episode_set", _boom, raising=False)
    train_samples, _ = make_group(episode="EP-train", split="train")
    heldout_samples, _ = make_group(episode="EP-heldout", split="heldout")
    samples = train_samples + heldout_samples
    units, heldout_episodes = trainer.build_sparse_history_units(samples)
    assert len(units) == 1
    kind, group, payload = units[0]
    assert kind == "sparse_group" and payload is None
    assert samples[group["SA"]]["episode"] == "EP-train"
    assert heldout_episodes == {"EP-heldout"}


def test_heldout_units_require_all_five_arms() -> None:
    complete, _ = make_group(episode="EP-h", split="heldout")
    groups = trainer.build_sparse_history_heldout_units(complete)
    assert set(groups) == {"EP-h:9"}
    assert set(groups["EP-h:9"]) == set(ARM_SPEC)

    missing_ra, _ = make_group(
        slots=tuple(slot for slot in ARM_SPEC if slot != "RA"),
        episode="EP-h",
        split="heldout",
    )
    with pytest.raises(ValueError):
        # 缺 RA 时主 claim SA-RA 无定义,必须报错而不是静默少算分母
        trainer.build_sparse_history_heldout_units(missing_ra)


def test_duplicate_arm_slot_in_one_group_is_rejected() -> None:
    samples, _ = make_group()
    samples.append(make_sample("SA"))
    with pytest.raises(ValueError):
        trainer.build_sparse_history_units(samples)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ({"schema_version": "causalcache.sparse_history_sample.v1"}, "schema 版本"),
        ({"reference_arm_id": "S0"}, "reference_arm_id 必须是 R0"),
        ({"deployment_baseline_arm_id": "R0"}, "deployment 基线必须是 N0"),
        ({"split": "validation"}, "未知 split"),
        ({"sparse": True}, "废弃字段 sparse"),
        ({"reference_variant": "native_recent2"}, "废弃字段 reference_variant"),
        ({"budget": 0}, "budget 必须为正"),
        ({"selected_steps": [8, 2]}, "selected_steps 必须严格递增"),
        ({"selected_steps": [2, 99]}, "选点必须早于 decision_step"),
        ({"adapter_mode": "active"}, "R0 的 adapter_mode 与契约不符"),
    ),
)
def test_sample_validation_is_fail_closed(
    mutation: dict[str, Any], reason: str
) -> None:
    sample = make_sample("R0")
    sample.update(mutation)
    with pytest.raises(ValueError):
        trainer.validate_sparse_sample(sample, index=0)


# ---------------------------------------------------------------------------
# 8. 数值梯度 vs 解析次梯度
# ---------------------------------------------------------------------------
def test_backward_weights_match_finite_differences() -> None:
    """有限差分校验:每个带梯度前向的 backward_weight * accumulation == dL/dℓ。

    # note (luojiaxuan): 两遍法把次梯度手算成标量权重再乘到各自前向上,写错符号
    # 或漏乘 share 都不会报错,只会静默训歪。取一组所有 hinge 都严格激活、Huber
    # 都在二次区且 drift 非零的点(远离折点),对每个臂做中心差分与解析权重比对。
    # 冻结锚点 ℓn⁰ 在扰动下保持不变 —— 这正是"drift 只对 active 分数求导"的定义。
    """
    accumulation = 3
    base = {
        "N0": 0.0, "R0": 0.0, "S0": 0.0, "RA": 0.0,
        "SA": -0.05,
        "SA_neg_step_shuffled": 0.0,
        "SA_neg_irrelevant": 0.02,
        "SA_neg_duplicate": -0.01,
    }
    frozen = {
        "SA_neg_step_shuffled": -0.03,
        "SA_neg_irrelevant": 0.05,
        "SA_neg_duplicate": 0.04,
    }
    samples, group = make_group()

    def loss_at(values: dict[str, float]) -> float:
        result = call_loss(
            samples, group, Recorder(values, frozen=frozen), accumulation=accumulation
        )
        assert result is not None
        return result

    recorder = Recorder(base, frozen=frozen)
    call_loss(samples, group, recorder, accumulation=accumulation)
    analytic = {
        slot: weight * accumulation for slot, weight in recorder.grad_weights.items()
    }
    assert set(analytic) == {"SA", *NEGATIVE_SLOTS}

    step = 1e-5
    for slot in analytic:
        plus = dict(base)
        plus[slot] += step
        minus = dict(base)
        minus[slot] -= step
        numeric = (loss_at(plus) - loss_at(minus)) / (2 * step)
        assert numeric == pytest.approx(analytic[slot], abs=1e-6), (
            f"{slot} 的解析次梯度与数值梯度不符"
        )


def test_backward_weight_is_divided_by_the_accumulation() -> None:
    samples, group = make_group()
    values = {slot: 0.0 for slot in group}
    values["SA"] = -0.05
    single = Recorder(values)
    call_loss(samples, group, single, accumulation=1)
    quad = Recorder(values)
    call_loss(samples, group, quad, accumulation=4)
    for slot, weight in single.grad_weights.items():
        assert quad.grad_weights[slot] == pytest.approx(weight / 4.0, abs=1e-12)


@requires_real_torch
def test_l2_penalty_enters_the_loss_and_the_gradient() -> None:
    import torch

    parameter = torch.nn.Parameter(torch.full((2,), 0.5))
    samples, group = make_group()
    values = {slot: 0.0 for slot in group}
    values["SA"] = 1.0  # 所有 hinge 都不激活、drift 为 0,剩下的只能是 L2
    result = call_loss(
        samples,
        group,
        Recorder(values),
        training={"sparse_history": True, "history_lora_l2_weight": 0.1},
        adapter_parameters=[parameter],
    )
    assert result == pytest.approx(0.1 * 2 * 0.25, abs=1e-9)
    assert parameter.grad is not None
    assert float(parameter.grad[0]) == pytest.approx(0.1 * 2 * 0.5, abs=1e-6)
