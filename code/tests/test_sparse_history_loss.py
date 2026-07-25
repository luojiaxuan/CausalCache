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
# 与 _sparse_history_group_loss_did 的 training.get(...) 缺省值逐字一致(= 预注册值)
SELECT_MARGIN = 0.01
GAIN_MARGIN = 0.01
CONTENT_MARGIN = 0.01
SELECT_WEIGHT = 1.0
GAIN_WEIGHT = 1.0
CONTENT_WEIGHT = 1.0
CAP_EPS = 0.02
CAP_WEIGHT = 2.0
# 弃用目标 legacy_sa_minus_r0 的缺省值,只有它的回归用例还用得到
LEGACY_RANK_MARGIN = 0.02
LEGACY_RANK_WEIGHT = 1.0
LEGACY_DRIFT_WEIGHT = 2.0
TRAINING = {"sparse_history": True, "history_lora_l2_weight": 0.0}
DID = trainer.SPARSE_OBJECTIVE_DID_RA_AWARE
LEGACY = trainer.SPARSE_OBJECTIVE_LEGACY


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
    objective_kind: str = DID,
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
        objective_kind=objective_kind,
        diagnostics_out=diagnostics_out,
    )


def did_components(
    samples: list[dict[str, Any]],
    group: dict[str, int],
    forward: Any,
    **kwargs: Any,
) -> dict[str, float]:
    """Run the DiD objective and return its per-term diagnostics.

    # note (luojiaxuan): 四条"不能再被见历史就放大刷高"的用例必须断言**分量**的方向,
    # 只断言总损失是不够的 —— 总损失下降既可能来自 select 改善,也可能来自 cap 罚金
    # 变小,两者的科学含义完全相反。
    """
    diagnostics: dict[str, float] = {}
    call_loss(samples, group, forward, diagnostics_out=diagnostics, **kwargs)
    return diagnostics


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
        "accumulation", "torch", "objective_kind", "diagnostics_out",
    }
    assert "wrapped" not in parameters, "LoRA 参数入参叫 adapter_parameters"
    assert parameters["diagnostics_out"].default is None
    # objective_kind 必须**没有**默认值:给了默认值等于让"忘了传"静默落到某一个目标上,
    # 而两个目标训出来的 adapter 语义完全不同(见 SPARSE_OBJECTIVE_KINDS 的注释)。
    assert parameters["objective_kind"].default is inspect.Parameter.empty


def test_unknown_objective_kind_is_fail_closed() -> None:
    samples, group = make_group()
    forward = Recorder({slot: 0.0 for slot in group})
    with pytest.raises(ValueError):
        call_loss(samples, group, forward, objective_kind="sa_minus_r0")


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
    # note (luojiaxuan): 白名单从 trainer 读,不在测试里硬编码 —— 损失新加一个分量诊断
    # (loss_select / loss_cap / did_content_*)时,两处各自漂移正是这条断言要防的事。
    runtime_only = set(trainer.SPARSE_RUNTIME_DIAGNOSTIC_KEYS)
    quantities = set(diagnostics) - runtime_only
    assert quantities <= set(trainer.SPARSE_GATE_VOCABULARY), (
        f"诊断键 {sorted(quantities - set(trainer.SPARSE_GATE_VOCABULARY))} 不在 gate 词表里"
    )
    for kind in trainer.SPARSE_NEGATIVE_KINDS:
        gap_key, drift_key = trainer.sparse_diagnostic_keys(kind)
        assert gap_key in diagnostics and drift_key in diagnostics
        # note (luojiaxuan): 2026-07-25 起负样本 margin 只报告、不作必需 gate ——
        # 预注册的 PASS 条件是 did_select / adapter_on_sparse / |A_r| / 三个 drift。
        # 内容敏感性由 L_content 在训练中优化,验收只看错误历史被 adapter 推动了多少。
        assert gap_key not in trainer.SPARSE_REQUIRED_GATES
        assert drift_key in trainer.SPARSE_REQUIRED_GATES
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


def test_identity_adapter_has_a_did_select_of_exactly_zero() -> None:
    """本次换目标的核心不变量:identity 上 A_c − A_r **恒等于 0**。

    # note (luojiaxuan): 旧目标 SA−R0 在 identity 上等于冻结模型自带的选点优势
    # S0−R0(实测 +0.0335),于是"什么都没学到"也能报出一个正的主 claim。差中差先对
    # 每条臂扣掉它自己的冻结基准,identity 上两个增量都是 0,差也就精确为 0 —— 这里
    # 故意把 S0/R0 放在互不相同、且离 0 很远的位置,证明恒等式与它们的位置无关。
    """
    samples, group = make_group()
    frozen_selection_advantage = 0.0335
    values = {
        "N0": -0.30,
        "R0": 0.11,
        "S0": 0.11 + frozen_selection_advantage,
        "RA": 0.11,  # identity: RA 与 R0 逐位相同
        "SA": 0.11 + frozen_selection_advantage,  # identity: SA 与 S0 逐位相同
        "SA_neg_step_shuffled": -0.20,
        "SA_neg_irrelevant": 0.40,
        "SA_neg_duplicate": 0.05,
    }
    diagnostics = did_components(
        samples, group, Recorder(values, frozen=dict(values))
    )
    assert diagnostics["adapter_on_sparse"] == 0.0
    assert diagnostics["adapter_on_recent"] == 0.0
    assert diagnostics["did_select"] == 0.0, (
        "identity 上差中差必须精确为 0,而不是约等于 0"
    )
    # 对照:同一份分数下旧目标的 SA−R0 直接把冻结模型的选点优势报成"能力"
    assert diagnostics["SA_minus_R0"] == pytest.approx(
        frozen_selection_advantage, abs=1e-12
    )
    # identity 上 drift-cap 一分不罚(三个 A_n 也都是 0),而三条 rank 项都还欠着 margin
    assert diagnostics["loss_cap"] == 0.0
    assert diagnostics["loss_select"] == pytest.approx(SELECT_MARGIN, abs=1e-12)
    assert diagnostics["loss_gain"] == pytest.approx(GAIN_MARGIN, abs=1e-12)
    assert diagnostics["loss_content"] == pytest.approx(CONTENT_MARGIN, abs=1e-12)


# --- 四条"不能再被『见历史就统一放大』刷高"的数值证据 -------------------------
# note (luojiaxuan): 共同基线是 identity(所有 A = 0)。每条用例只动一处 active 分数,
# 断言的是**分量的方向**:只断言总损失会把"select 改善"与"cap 罚金变小"混为一谈,
# 而这两件事的科学含义正好相反。
_IDENTITY_VALUES = {slot: 0.0 for slot in ARM_SPEC}


def _did_at(**deltas: float) -> dict[str, float]:
    """DiD diagnostics when the listed active arms move off identity by ``delta``."""
    samples, group = make_group()
    active = dict(_IDENTITY_VALUES)
    for slot, delta in deltas.items():
        active[slot] = active[slot] + delta
    # frozen 端恒定在 identity:bypass 分数与 adapter 参数无关,这正是 A 的定义
    return did_components(
        samples, group, Recorder(active, frozen=dict(_IDENTITY_VALUES))
    )


IDENTITY = _did_at()


def test_uniform_amplification_of_every_active_arm_buys_nothing() -> None:
    """(1) 所有 active 臂共同 +0.1:select/content 一分不改善,drift-cap 必须罚。"""
    shifted = _did_at(SA=0.1, RA=0.1, **{slot: 0.1 for slot in NEGATIVE_SLOTS})
    # 三个增量一起动 → 差中差与 content 差全部原地不动
    assert shifted["did_select"] == pytest.approx(IDENTITY["did_select"], abs=1e-12)
    assert shifted["loss_select"] == pytest.approx(IDENTITY["loss_select"], abs=1e-12)
    assert shifted["loss_content"] == pytest.approx(
        IDENTITY["loss_content"], abs=1e-12
    )
    for kind in trainer.SPARSE_NEGATIVE_KINDS:
        key = trainer.sparse_content_diagnostic_key(kind)
        assert shifted[key] == pytest.approx(IDENTITY[key], abs=1e-12)
    # 唯一动了的是 gain(A_c 真的涨了 0.1)与 cap(A_r 与三个 A_n 全部越界)
    assert shifted["loss_gain"] < IDENTITY["loss_gain"]
    expected_cap = CAP_WEIGHT * (0.1 - CAP_EPS) + CAP_WEIGHT * 1.0 * (0.1 - CAP_EPS)
    assert IDENTITY["loss_cap"] == 0.0
    assert shifted["loss_cap"] == pytest.approx(expected_cap, abs=1e-12)
    assert shifted["loss_cap"] > IDENTITY["loss_cap"], "统一放大必须被 drift-cap 罚"


def test_lifting_only_the_sparse_arm_improves_every_rank_term() -> None:
    """(2) 只有 SA +0.02:select / gain / content 全部改善,cap 一分不罚。"""
    shifted = _did_at(SA=0.02)
    assert shifted["did_select"] == pytest.approx(0.02, abs=1e-12)
    assert shifted["adapter_on_sparse"] == pytest.approx(0.02, abs=1e-12)
    # 0.02 已越过全部三个 margin(预注册值都是 0.01),三个 hinge 一起落到 0
    assert shifted["loss_select"] == pytest.approx(max(SELECT_MARGIN - 0.02, 0.0))
    assert shifted["loss_gain"] == pytest.approx(max(GAIN_MARGIN - 0.02, 0.0))
    assert shifted["loss_content"] == pytest.approx(max(CONTENT_MARGIN - 0.02, 0.0))
    assert shifted["loss_select"] < IDENTITY["loss_select"]
    assert shifted["loss_gain"] < IDENTITY["loss_gain"]
    assert shifted["loss_content"] < IDENTITY["loss_content"]
    # 半步(+0.005,margin 之内)时三项按 1:1 线性改善,证明改善不是"一次性跳到 0"
    half = _did_at(SA=0.005)
    assert half["loss_select"] == pytest.approx(SELECT_MARGIN - 0.005, abs=1e-12)
    assert half["loss_gain"] == pytest.approx(GAIN_MARGIN - 0.005, abs=1e-12)
    assert half["loss_content"] == pytest.approx(CONTENT_MARGIN - 0.005, abs=1e-12)
    # A_r 与三个 A_n 都还停在 0,dead zone 内一分不罚
    assert shifted["loss_cap"] == 0.0


def test_lifting_only_the_recent_arm_makes_selection_worse() -> None:
    """(3) 只有 RA +0.02:select 变差,且 drift-cap 不再是 0(|A_r| 已到边界外)。"""
    shifted = _did_at(RA=0.02)
    assert shifted["did_select"] == pytest.approx(-0.02, abs=1e-12)
    assert shifted["loss_select"] == pytest.approx(SELECT_MARGIN + 0.02, abs=1e-12)
    assert shifted["loss_select"] > IDENTITY["loss_select"], (
        "抬 recent 臂必须让 select 项变差 —— 旧目标里它是免费的"
    )
    # gain / content 只看 A_c,与 RA 无关
    assert shifted["loss_gain"] == pytest.approx(IDENTITY["loss_gain"], abs=1e-12)
    assert shifted["loss_content"] == pytest.approx(
        IDENTITY["loss_content"], abs=1e-12
    )
    # note (luojiaxuan): 预注册的 eps 恰好也是 0.02,所以 +0.02 落在 dead zone 的**折点
    # 上** —— [|A_r| − eps]+ 在这里精确为 0(罚金还没开始,但已到激活边界)。再多一点点
    # 就必须按 λ=2 的斜率收罚;这一条同时钉住"边界不罚"与"越界即罚"两侧。
    assert abs(shifted["adapter_on_recent"]) == pytest.approx(CAP_EPS, abs=1e-12)
    assert shifted["loss_cap"] == 0.0
    over = _did_at(RA=0.021)
    assert over["loss_cap"] == pytest.approx(CAP_WEIGHT * (0.021 - CAP_EPS), abs=1e-12)
    assert over["loss_cap"] > IDENTITY["loss_cap"], "越过 eps 之后 drift-cap 必须激活"


def test_lifting_only_a_negative_makes_content_worse() -> None:
    """(4) 只有 negative +0.02:content 变差,且越过 eps 后 drift-cap 激活。"""
    shifted = _did_at(SA_neg_irrelevant=0.02)
    share = 1.0 / (1.0 + 1.0 + 0.5)
    content_key = trainer.sparse_content_diagnostic_key("irrelevant")
    assert shifted[content_key] == pytest.approx(-0.02, abs=1e-12)
    assert shifted["loss_content"] == pytest.approx(
        IDENTITY["loss_content"] + share * 0.02, abs=1e-12
    )
    assert shifted["loss_content"] > IDENTITY["loss_content"], (
        "抬错误历史必须让 content 项变差"
    )
    # select / gain 只看 A_c 与 A_r,与负样本无关
    assert shifted["loss_select"] == pytest.approx(IDENTITY["loss_select"], abs=1e-12)
    assert shifted["loss_gain"] == pytest.approx(IDENTITY["loss_gain"], abs=1e-12)
    # 同上:+0.02 恰好落在 dead zone 的折点,罚金精确为 0 而边界已经到达
    assert shifted[trainer.sparse_diagnostic_keys("irrelevant")[1]] == pytest.approx(
        CAP_EPS, abs=1e-12
    )
    assert shifted["loss_cap"] == 0.0
    over = _did_at(SA_neg_irrelevant=0.021)
    assert over["loss_cap"] == pytest.approx(
        CAP_WEIGHT * share * (0.021 - CAP_EPS), abs=1e-12
    )
    assert over["loss_cap"] > IDENTITY["loss_cap"], "越过 eps 之后 drift-cap 必须激活"


def test_the_did_terms_add_up() -> None:
    """总损失恒等于四个分量之和(加 L2),分量诊断不是另算一套。"""
    samples, group = make_group()
    values = {slot: 0.0 for slot in group}
    values["SA"] = -0.05
    forward = Recorder(values)
    diagnostics: dict[str, float] = {}
    result = call_loss(samples, group, forward, diagnostics_out=diagnostics)
    expected = (
        SELECT_WEIGHT * (SELECT_MARGIN + 0.05)
        + GAIN_WEIGHT * (GAIN_MARGIN + 0.05)
        + CONTENT_WEIGHT * (CONTENT_MARGIN + 0.05)
    )
    assert result == pytest.approx(expected, abs=1e-9)
    assert diagnostics["loss"] == pytest.approx(
        diagnostics["loss_select"]
        + diagnostics["loss_gain"]
        + diagnostics["loss_content"]
        + diagnostics["loss_cap"]
        + diagnostics["loss_l2"],
        abs=1e-12,
    )
    assert forward.grad_weights["SA"] < 0.0, "A_c 落后时梯度必须推高 ℓ_SA"
    assert forward.grad_weights["RA"] > 0.0, (
        "select hinge 激活时梯度必须压低 ℓ_RA —— 但 L_gain 挡住了『只压 RA』的捷径"
    )


def test_legacy_objective_still_reproduces_its_gain_and_rank_hinges() -> None:
    """弃用目标保持逐字不变:旧 checkpoint 的复现依赖它。"""
    samples, group = make_group()
    values = {slot: 0.0 for slot in group}
    values["SA"] = -0.05
    forward = Recorder(values)
    result = call_loss(samples, group, forward, objective_kind=LEGACY)
    expected = GAIN_WEIGHT * (GAIN_MARGIN + 0.05) + LEGACY_RANK_WEIGHT * (
        LEGACY_RANK_MARGIN + 0.05
    )
    assert result == pytest.approx(expected, abs=1e-9)
    assert forward.grad_weights["SA"] < 0.0
    for slot in ("N0", "S0", "RA"):
        assert not forward.touched(slot), f"legacy 目标不读 {slot}"


def test_legacy_objective_still_uses_the_smooth_l1_drift() -> None:
    """弃用目标的 drift 仍是 SmoothL1 —— 换成 dead-zone hinge 只发生在新目标里。

    # note (luojiaxuan): 这条同时是"为什么必须换"的对照。同一个 A_n = 0.06,
    # SmoothL1 的梯度是 2*0.06 = 0.12,而 rank hinge 的梯度量级是 1;新目标的
    # dead-zone hinge 在同一点给出 2.0(见 test_cap_penalises_only_the_adapter_...)。
    """
    samples, group = make_group(slots=("R0", "SA", "SA_neg_irrelevant"))
    drift = 0.06
    forward = Recorder(
        {"R0": 0.0, "SA": 0.5, "SA_neg_irrelevant": drift},
        frozen={"SA_neg_irrelevant": 0.0},
    )
    result = call_loss(samples, group, forward, objective_kind=LEGACY)
    assert result == pytest.approx(
        LEGACY_DRIFT_WEIGHT * 1.0 * 0.5 * drift * drift, abs=1e-12
    )
    assert forward.grad_weights["SA_neg_irrelevant"] == pytest.approx(
        LEGACY_DRIFT_WEIGHT * drift, abs=1e-12
    )
    assert forward.grad_weights["SA_neg_irrelevant"] < 1.0, (
        "SmoothL1 在 drift≈0.06 处的梯度只有 0.12,压不住量级为 1 的 rank hinge —— "
        "这是损失尺度决定的,不是训练偶然性"
    )


def test_negative_shares_are_normalised_over_the_present_negatives() -> None:
    """duplicate 单独成组时份额是 1.0,不是它的原始 scale 0.5。

    # note (luojiaxuan): 归一化的分母只能是**本组真正进入损失**的负样本。按固定
    # scale 求和会让 K=1 组(只剩 irrelevant)的负样本质量只有别组的 40%,等于按
    # 预算给梯度加权;按 NEGATIVE_SPECS 的全集归一化则更糟(份额 0.2)。这条用例
    # 同时把这两种写法钉死。did 目标需要 R0/S0/RA/SA 四臂齐全,所以组比旧版大三条臂,
    # 但被考察的仍然只是负样本份额。
    """
    samples, group = make_group(slots=("R0", "S0", "RA", "SA", "SA_neg_duplicate"))
    forward = Recorder({slot: 0.0 for slot in group})
    result = call_loss(samples, group, forward)
    assert result == pytest.approx(
        SELECT_MARGIN + GAIN_MARGIN + 1.0 * CONTENT_MARGIN, abs=1e-9
    )
    assert forward.grad_weights["SA_neg_duplicate"] == pytest.approx(1.0, abs=1e-12)


def test_k1_inventory_keeps_the_full_negative_mass() -> None:
    """K=1 组只剩 irrelevant(shuffled/duplicate 与 SA 逐字相同,不入库)。"""
    samples, group = make_group(
        slots=("N0", "R0", "S0", "RA", "SA", "SA_neg_irrelevant"), budget=1
    )
    forward = Recorder({slot: 0.0 for slot in group})
    result = call_loss(samples, group, forward)
    assert result == pytest.approx(
        SELECT_MARGIN + GAIN_MARGIN + 1.0 * CONTENT_MARGIN, abs=1e-9
    )
    assert forward.grad_weights["SA_neg_irrelevant"] == pytest.approx(1.0, abs=1e-12)


def test_negative_scale_is_read_from_the_field_not_a_name_table() -> None:
    """把 duplicate 的 scale 改成 0.25,份额必须跟着变;不变说明在查名字硬编码表。"""
    slots = ("R0", "S0", "RA", "SA", "SA_neg_duplicate", "SA_neg_irrelevant")
    values = {slot: 0.0 for slot in slots}
    # duplicate 被 adapter 压低 0.05:content hinge 不再激活(A_c - A_n = +0.05),
    # 而 |A_n| = 0.05 > eps 让 drift-cap 激活 —— 两个份额出口都被这条用例看到。
    frozen = {"SA_neg_duplicate": 0.05}

    default_samples, default_group = make_group(slots=slots)
    default_forward = Recorder(values, frozen=frozen)
    default_result = call_loss(default_samples, default_group, default_forward)
    # scale_mass = 0.5 + 1.0
    default_expected = (
        SELECT_MARGIN
        + GAIN_MARGIN
        + (1.0 / 1.5) * CONTENT_MARGIN
        + CAP_WEIGHT * (0.5 / 1.5) * (0.05 - CAP_EPS)
    )
    assert default_result == pytest.approx(default_expected, abs=1e-9)
    assert default_forward.grad_weights["SA_neg_irrelevant"] == pytest.approx(
        1.0 / 1.5, abs=1e-12
    )

    retuned_samples, retuned_group = make_group(
        slots=slots, overrides={"SA_neg_duplicate": {"negative_scale": 0.25}}
    )
    retuned_forward = Recorder(values, frozen=frozen)
    retuned_result = call_loss(retuned_samples, retuned_group, retuned_forward)
    retuned_expected = (
        SELECT_MARGIN
        + GAIN_MARGIN
        + (1.0 / 1.25) * CONTENT_MARGIN
        + CAP_WEIGHT * (0.25 / 1.25) * (0.05 - CAP_EPS)
    )
    assert retuned_result == pytest.approx(retuned_expected, abs=1e-9)
    assert retuned_forward.grad_weights["SA_neg_irrelevant"] == pytest.approx(
        1.0 / 1.25, abs=1e-12
    )
    assert retuned_result != pytest.approx(default_result, abs=1e-9)


def test_negative_membership_comes_from_role_not_slot_name() -> None:
    """诊断字段 variant 写成负样本名、role 却是 measurement 的臂不得进 content/cap。"""
    decoy = make_sample("S0", variant="sparse_step_shuffled", negative_scale=1.0)
    decoy["arm_slot"] = "sparse_step_shuffled"
    samples, group = make_group(slots=("R0", "S0", "RA", "SA"), extra=(decoy,))
    values = {slot: 0.0 for slot in group}
    values["sparse_step_shuffled"] = -5.0
    forward = Recorder(values)
    diagnostics: dict[str, float] = {}
    result = call_loss(samples, group, forward, diagnostics_out=diagnostics)
    assert result == pytest.approx(SELECT_MARGIN + GAIN_MARGIN, abs=1e-9)
    assert diagnostics["negatives"] == 0.0
    assert not forward.touched("sparse_step_shuffled"), (
        "role=measurement 的臂不得因为 variant/arm_slot 长得像负样本就进损失"
    )


def test_only_the_deployment_baseline_stays_out_of_the_training_loss() -> None:
    """did 目标读 R0/S0/RA/SA 四臂,只有部署基线 N0 仍然只在留出集上打分。

    # note (luojiaxuan): 取代旧的 test_measurement_arms_stay_out_of_the_training_loss。
    # 那条断言 S0/RA 一律不进训练损失 —— 那正是旧目标的定义,而"训练时看不见 RA"
    # 恰恰是它把『见历史就放大』当成能力的原因。新目标必须读这两条臂:S0 是 A_c 的
    # 基准端,RA 是 A_r 的 active 端。这里同时钉住"读哪些"与"仍然不读哪一条"。
    """
    samples, group = make_group()
    values = {slot: 0.0 for slot in group}
    values["SA"] = -0.05
    forward = Recorder(values)
    call_loss(samples, group, forward)
    assert not forward.touched("N0"), "N0 只在留出集打分,不进训练损失"
    for slot in ("R0", "S0", "RA", "SA"):
        assert forward.touched(slot), f"did 目标必须读 {slot}"
    assert trainer.sparse_objective_excluded_arms(DID) == ["N0"]
    assert trainer.sparse_objective_excluded_arms(LEGACY) == ["N0", "RA", "S0"]


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
    for function in (
        trainer._sparse_history_group_loss,
        trainer._sparse_history_group_loss_did,
        trainer._sparse_history_group_loss_legacy,
    ):
        source = inspect.getsource(function)
        assert "ce_weight" not in source, "sparse 目标的 CE 权重恒为 0,不得再读 ce 权重"


# ---------------------------------------------------------------------------
# 4. drift-cap:锚到**自身冻结分数**的 dead-zone hinge(取代旧的 SmoothL1)
# ---------------------------------------------------------------------------
def test_cap_anchors_each_negative_to_its_own_frozen_score() -> None:
    """负样本停在自己的冻结分数上就不该受罚,哪怕它离 R0 很远。

    # note (luojiaxuan): 锚点语义(锚到自身 bypass 分数而不是锚到 R0)在换目标时**没有
    # 变**,变的只是罚函数的形状:SmoothL1 → dead-zone hinge。这条用例仍然只考察锚点。
    """
    samples, group = make_group(slots=("R0", "S0", "RA", "SA", "SA_neg_irrelevant"))
    # 冻结模型本来就偏好这条负样本(0.3 ≫ R0 的 0.0),但 adapter 没有动它
    forward = Recorder(
        {"R0": 0.0, "S0": 0.0, "RA": 0.0, "SA": 0.5, "SA_neg_irrelevant": 0.3},
        frozen={"SA_neg_irrelevant": 0.3},
    )
    assert call_loss(samples, group, forward) is None, (
        "A_n = 0 时 cap 不罚;旧的锚到 R0 语义会在这里收一笔与 adapter 无关的罚"
    )
    assert forward.grad_weights == {}


def test_cap_penalises_only_the_adapter_induced_displacement() -> None:
    """dead-zone hinge:超出 eps 的部分按 λ 线性收罚,梯度量级恒为 λ。

    # note (luojiaxuan): 这条同时是"为什么不能继续用 SmoothL1"的数值证据。同一个
    # A_n = 0.3 下,旧的 SmoothL1(权重 2)给出的梯度是 2*0.3 = 0.6,而 A_n 落在真实
    # 观测的 0.06 量级时只有 0.12 —— 比 rank hinge 的 1 小一个量级,anchor 压不住是
    # **损失尺度本身决定的**。dead-zone hinge 的梯度与 A_n 的大小无关,恒为 λ = 2。
    """
    samples, group = make_group(slots=("R0", "S0", "RA", "SA", "SA_neg_irrelevant"))
    forward = Recorder(
        {"R0": 0.0, "S0": 0.0, "RA": 0.0, "SA": 0.5, "SA_neg_irrelevant": 0.3},
        frozen={"SA_neg_irrelevant": 0.0},
    )
    result = call_loss(samples, group, forward)
    # A_c = 0.5 → select/gain 都不激活;content 差 0.5-0.3 = 0.2 也不激活;只剩 cap
    assert result == pytest.approx(CAP_WEIGHT * 1.0 * (0.3 - CAP_EPS), abs=1e-9)
    assert forward.grad_weights["SA_neg_irrelevant"] == pytest.approx(
        CAP_WEIGHT, abs=1e-12
    ), "梯度必须把被 adapter 抬高的负样本压回它自己的冻结分数,且量级与 rank 项同阶"

    smaller = Recorder(
        {"R0": 0.0, "S0": 0.0, "RA": 0.0, "SA": 0.5, "SA_neg_irrelevant": 0.06},
        frozen={"SA_neg_irrelevant": 0.0},
    )
    call_loss(samples, group, smaller)
    assert smaller.grad_weights["SA_neg_irrelevant"] == pytest.approx(
        CAP_WEIGHT, abs=1e-12
    ), "drift 缩到 0.06 时 SmoothL1 只剩 0.12 的梯度,dead-zone hinge 仍然是 2"


def test_cap_is_independent_of_where_the_frozen_reference_sits() -> None:
    """把 R0 与 RA 一起挪走(A_r 不变),整条损失一分不变。

    # note (luojiaxuan): 取代旧的 test_drift_is_independent_of_where_the_reference_sits。
    # 旧版只挪 R0 并断言损失不变 —— 那在 did 目标下**必然失败,而且是应该失败**:
    # R0 是 A_r 的冻结基准端,只挪它就等于人为制造一个 adapter 位移。新版把 recent 臂
    # 的两端一起挪(A_r 恒定),考察的仍是"负样本的 cap 与参考臂坐在哪无关"。
    """
    samples, group = make_group(slots=("R0", "S0", "RA", "SA", "SA_neg_irrelevant"))
    frozen = {"SA_neg_irrelevant": 0.0}
    near = call_loss(
        samples, group,
        Recorder(
            {"R0": 0.0, "S0": 0.0, "RA": 0.0, "SA": 0.5, "SA_neg_irrelevant": 0.3},
            frozen=frozen,
        ),
    )
    far = call_loss(
        samples, group,
        Recorder(
            {"R0": -0.4, "S0": 0.0, "RA": -0.4, "SA": 0.5, "SA_neg_irrelevant": 0.3},
            frozen=frozen,
        ),
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
        if slot in ("SA", "RA", "R0", "S0"):
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
    expected = (
        SELECT_WEIGHT * (SELECT_MARGIN + 0.05)
        + GAIN_WEIGHT * (GAIN_MARGIN + 0.05)
        + CONTENT_WEIGHT * (CONTENT_MARGIN + 0.05)
    )
    assert result == pytest.approx(expected, abs=1e-9)
    assert "SA_neg_duplicate" not in forward.grad_weights


# ---------------------------------------------------------------------------
# 5. reference 的选择:读 reference_arm_id 字段,不猜前缀
# ---------------------------------------------------------------------------
def test_reference_is_taken_from_the_reference_arm_id_field() -> None:
    decoy = make_sample("S0", variant="native_recent2")
    decoy["arm_slot"] = "native_recent2"
    samples, group = make_group(slots=("R0", "S0", "RA", "SA"), extra=(decoy,))
    forward = Recorder(
        {"R0": 0.0, "S0": 0.0, "RA": 0.0, "SA": -0.05, "native_recent2": 10.0}
    )
    result = call_loss(samples, group, forward)
    assert forward.touched("R0")
    assert not forward.touched("native_recent2"), (
        "参考臂必须来自 reference_arm_id=R0,不得因为名字像 native_recent 就被选中"
    )
    assert result == pytest.approx(
        (SELECT_MARGIN + 0.05) + (GAIN_MARGIN + 0.05), abs=1e-9
    )


def test_frozen_baseline_forwards_never_carry_gradient() -> None:
    """A_c 与 A_r 的 bypass 端(S0 / R0)都是冻结量,只能 no-grad 取值。

    # note (luojiaxuan): 由 test_reference_forward_never_carries_gradient 扩写。did
    # 目标多了一条冻结基准 S0,它与 R0 同为"跑在冻结 policy 上、与 adapter 参数无关"
    # 的量 —— 两者都必须走冻结分数缓存,任何一次带梯度前向都是实现错误。
    """
    samples, group = make_group()
    values = {slot: 0.0 for slot in group}
    values["SA"] = -0.05
    forward = Recorder(values)
    call_loss(samples, group, forward)
    for slot in ("R0", "S0"):
        assert all(
            not grad for called, grad, _weight, _mode in forward.calls
            if called == slot
        ), f"{slot} 是冻结基准,只能 no-grad 取值"
        assert slot not in forward.grad_weights
    # 对照:RA 的 active 端**必须**有带梯度前向(L_select 与 L_cap 都要对它求导)
    assert "RA" in forward.grad_weights


def test_inconsistent_reference_arm_id_is_rejected() -> None:
    samples, group = make_group(
        overrides={"SA_neg_irrelevant": {"reference_arm_id": "N0"}}
    )
    forward = Recorder({slot: 0.0 for slot in group})
    with pytest.raises(ValueError):
        call_loss(samples, group, forward)


@pytest.mark.parametrize("dropped", ("R0", "S0", "RA", "SA"))
def test_missing_required_arm_is_fail_closed(dropped: str) -> None:
    """did 目标要求 R0/S0/RA/SA 四臂齐全,缺任一臂抛错而不是静默跳过。

    # note (luojiaxuan): 由 ("R0", "SA") 扩到四臂。缺 S0 则 A_c 无定义、缺 RA 则 A_r
    # 无定义,而"差中差在 identity 上恒为 0"正是靠这两个基准端构造出来的 —— 静默
    # return None 会让主 claim 的分母悄悄变小,stdout 上只表现为组数变少(审计第 8 条)。
    """
    slots = tuple(slot for slot in ARM_SPEC if slot != dropped)
    samples, group = make_group(slots=slots)
    forward = Recorder({slot: 0.0 for slot in group})
    with pytest.raises((ValueError, KeyError)):
        call_loss(samples, group, forward)
    assert not forward.touched(dropped)


@pytest.mark.parametrize("dropped", ("S0", "RA"))
def test_unit_builder_rejects_a_group_the_objective_cannot_score(dropped: str) -> None:
    """四臂检查也在单元构造期跑一次:训练 150 步之后才发现缺臂已经太晚。"""
    slots = tuple(slot for slot in ARM_SPEC if slot != dropped)
    samples, _ = make_group(slots=slots)
    with pytest.raises(ValueError):
        trainer.build_sparse_history_units(samples, objective_kind=DID)
    # 弃用目标不读这两条臂,同一份语料对它仍然合法
    units, _ = trainer.build_sparse_history_units(samples, objective_kind=LEGACY)
    assert len(units) == 1


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
def test_unit_builder_takes_samples_and_the_objective_kind_only() -> None:
    """# note (luojiaxuan): 由 test_unit_builder_signature_takes_only_samples 改写。

    原意是"split 的权威只有样本的 split 字段,构造期不接受 training= 之类的旁路",
    这一条**没有放松**:下面仍然显式断言不存在 training 形参。新增的 objective_kind
    只决定"这个目标需要哪些臂齐全",不参与分组、不参与 split 判定。
    """
    parameters = inspect.signature(trainer.build_sparse_history_units).parameters
    assert list(parameters) == ["samples", "objective_kind"]
    assert "training" not in parameters, (
        "split 的权威只有样本的 split 字段;单元构造不接受 training= 之类的旁路"
    )
    assert parameters["objective_kind"].kind is inspect.Parameter.KEYWORD_ONLY


def test_split_comes_from_the_sample_field(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("trainer 不得重算 heldout hash;split 以样本字段为准")

    monkeypatch.setattr(trainer, "heldout_episode_set", _boom, raising=False)
    train_samples, _ = make_group(episode="EP-train", split="train")
    heldout_samples, _ = make_group(episode="EP-heldout", split="heldout")
    samples = train_samples + heldout_samples
    units, heldout_episodes = trainer.build_sparse_history_units(
        samples, objective_kind=DID
    )
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
        trainer.build_sparse_history_units(samples, objective_kind=DID)


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
@pytest.mark.parametrize("objective_kind", (DID, LEGACY))
def test_backward_weights_match_finite_differences(objective_kind: str) -> None:
    """有限差分校验:每个带梯度前向的 backward_weight * accumulation == dL/dℓ。

    # note (luojiaxuan): 两遍法把次梯度手算成标量权重再乘到各自前向上,写错符号
    # 或漏乘 share 都不会报错,只会静默训歪。取一组所有 hinge 与 dead zone 都**严格**
    # 激活的点(远离每一处折点),对每个可导臂做中心差分与解析权重比对。
    # 可导量只有 active 端 ℓ_SA / ℓ_RA / ℓ_n_active;冻结端 ℓ_S0 / ℓ_R0 / ℓ_n_bypass
    # 在扰动下保持不变 —— 这正是"只对 A 的 active 端求导"的定义。
    # 选点(did):A_c = -0.05、A_r = +0.10、A_n = +0.03 / -0.03 / -0.05,四个 cap
    # 全部越过 eps=0.02 且符号有正有负,三个 rank hinge 全部严格为正。
    """
    accumulation = 3
    base = {
        "N0": 0.0, "R0": 0.0, "S0": 0.0,
        "RA": 0.10,
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
            samples,
            group,
            Recorder(values, frozen=frozen),
            accumulation=accumulation,
            objective_kind=objective_kind,
        )
        assert result is not None
        return result

    recorder = Recorder(base, frozen=frozen)
    call_loss(
        samples,
        group,
        recorder,
        accumulation=accumulation,
        objective_kind=objective_kind,
    )
    analytic = {
        slot: weight * accumulation for slot, weight in recorder.grad_weights.items()
    }
    expected_slots = {"SA", *NEGATIVE_SLOTS}
    if objective_kind == DID:
        expected_slots.add("RA")
    assert set(analytic) == expected_slots

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


# ---------------------------------------------------------------------------
# 9. 目标开关:config 段、消费的超参、以及调度器的前向次数成本模型
# ---------------------------------------------------------------------------
def _sparse_config() -> dict[str, Any]:
    path = _CODE_ROOT / "configs" / "causalcache_sparse_history_v1.json"
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def test_the_shipped_config_selects_the_did_objective() -> None:
    config = _sparse_config()
    kind, objective = trainer.validate_sparse_objective(config)
    assert kind == DID
    assert objective["excluded_arms"] == ["N0"], (
        "did 目标要读 RA(A_r 的 active 端),RA 必须从 excluded_arms 里移除"
    )
    # 预注册超参逐项对照,配错一个数就是另一个实验
    training = config["training"]
    assert training["sparse_select_margin"] == 0.01
    assert training["sparse_gain_margin"] == 0.01
    assert training["sparse_content_margin"] == 0.01
    assert training["sparse_select_weight"] == 1.0
    assert training["sparse_gain_weight"] == 1.0
    assert training["sparse_content_weight"] == 1.0
    assert training["sparse_drift_cap_eps"] == 0.02
    assert training["sparse_drift_cap_weight"] == 2.0
    assert training["history_ce_weight"] == 0.0


def test_the_shipped_config_training_block_has_no_unconsumed_key() -> None:
    config = _sparse_config()
    kind, _objective = trainer.validate_sparse_objective(config)
    consumed = set(trainer.sparse_training_keys(kind))
    assert set(config["training"]) == consumed
    # 弃用目标的 rank/drift 超参不得残留在 did 的 config 里(写了却不生效)
    for retired in ("sparse_rank_margin", "sparse_rank_weight", "sparse_drift_weight"):
        assert retired not in config["training"]


def test_objective_block_is_fail_closed() -> None:
    config = _sparse_config()

    missing_kind = {**config, "objective": {
        key: value for key, value in config["objective"].items() if key != "kind"
    }}
    with pytest.raises(ValueError):
        trainer.validate_sparse_objective(missing_kind)

    unknown_kind = {**config, "objective": {**config["objective"], "kind": "whatever"}}
    with pytest.raises(ValueError):
        trainer.validate_sparse_objective(unknown_kind)

    stray_key = {**config, "objective": {**config["objective"], "L_rank": "..."}}
    with pytest.raises(ValueError):
        trainer.validate_sparse_objective(stray_key)

    # 核心结构性检查:did 目标真的在读 RA,excluded_arms 里再写 RA 就必须报错
    stale_exclusion = {**config, "objective": {
        **config["objective"], "excluded_arms": ["N0", "RA", "S0"],
    }}
    with pytest.raises(ValueError):
        trainer.validate_sparse_objective(stale_exclusion)


def test_training_keys_are_partitioned_by_objective() -> None:
    did_keys = set(trainer.sparse_training_keys(DID))
    legacy_keys = set(trainer.sparse_training_keys(LEGACY))
    assert {"sparse_select_margin", "sparse_content_weight",
            "sparse_drift_cap_eps", "sparse_drift_cap_weight"} <= did_keys
    assert {"sparse_rank_margin", "sparse_rank_weight",
            "sparse_drift_weight"} <= legacy_keys
    # 不取并集:给 did 写 rank 超参会以 unconsumed 报错,反之亦然
    assert did_keys & {"sparse_rank_margin", "sparse_rank_weight",
                       "sparse_drift_weight"} == set()
    assert legacy_keys & {"sparse_select_margin", "sparse_content_margin",
                          "sparse_drift_cap_eps"} == set()
    with pytest.raises(ValueError):
        trainer.sparse_training_keys("whatever")


@pytest.mark.parametrize(
    ("budget", "slots", "did_forwards", "legacy_forwards"),
    (
        (2, tuple(ARM_SPEC), 15, 12),
        (1, ("N0", "R0", "S0", "RA", "SA", "SA_neg_irrelevant"), 9, 6),
    ),
)
def test_schedule_cost_counts_the_did_forwards(
    budget: int, slots: tuple[str, ...], did_forwards: int, legacy_forwards: int
) -> None:
    """K≥2 组从 12 次前向涨到 15 次:第一遍多 S0/RA,第二遍多 RA 的带梯度前向。"""
    samples, group = make_group(slots=slots, budget=budget)
    assert trainer.sparse_group_schedule_cost(
        samples, group, objective_kind=DID
    ) == (did_forwards, budget)
    assert trainer.sparse_group_schedule_cost(
        samples, group, objective_kind=LEGACY
    ) == (legacy_forwards, budget)
    with pytest.raises(ValueError):
        trainer.sparse_group_schedule_cost(samples, group, objective_kind="whatever")


def test_did_select_is_a_derived_quantity_of_the_five_arms() -> None:
    """训练目标优化的量必须能在留出集上按同一份代数式复算。"""
    assert trainer.SPARSE_DERIVED_QUANTITIES["did_select"] == "(SA - S0) - (RA - R0)"
    assert "did_select" in trainer.SPARSE_GATE_VOCABULARY
    scores = {"N0": -0.3, "R0": 0.11, "S0": 0.1435, "RA": 0.13, "SA": 0.17}
    value = trainer.evaluate_gate_expression(
        trainer.SPARSE_DERIVED_QUANTITIES["did_select"], scores
    )
    assert value == pytest.approx((0.17 - 0.1435) - (0.13 - 0.11), abs=1e-12)
    # note (luojiaxuan): 2026-07-25 验收标准已随目标一并换成 RA-aware 判据。
    # did_select 现在既是主判据也是 composite;而 SA_minus_R0 必须**退出** must_pass ——
    # 新目标不优化它,留着会把选点拉向"见历史就放大"最严重的 checkpoint
    # (v5 实测 SA_minus_R0 从 +0.034 涨到 +0.118,而真实的 SA-RA 归零)。
    gates = _sparse_config()["gates"]
    assert gates["must_pass"]["did_select"] == "> 0 @ci_low"
    assert gates["composite_score"] == "did_select"
    assert "SA_minus_R0" not in gates["must_pass"]
    assert "SA_minus_R0" not in gates["composite_score"]
    # A_r 必须被封顶,否则 did_select 可以靠压低 RA 而不是抬高 SA 来刷高
    assert gates["must_pass"]["adapter_on_recent_abs"] == "< 0.02"
