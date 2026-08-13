#!/usr/bin/env python3
"""Phase 1 最后一条验收:**不同 memory subset 确实改变任务成功概率**(GPU harness)。

# note (luojiaxuan): 路线见 rl/docs/agentic_memory_rl_roadmap_20260813.md 决策 E5。
# 本脚本用**冻结** GUI-Owl-1.5-8B 在 causalcache_agentic 程序化环境里跑闭环
# rollout,同一批任务分别用四条固定记忆规则(臂),比较终局任务成功率:
#   * ``recent2``  —— 默认基线,最近 B 个合法历史帧;
#   * ``oracle``   —— 用 ``memory_probe["required_steps"]``(诊断特权,不足补 recent);
#   * ``random2``  —— 候选集上均匀采样 B 张(sanity 臂,seed 固定可复现);
#   * ``none``     —— B=0,完全不给历史保留帧。
# 奖励口径:**只有终局 verifier 的 0/1** —— 不读 gold 动作、不算 milestone、不做
# step-level 正确性(路线 §9 明令禁止)。``memory_probe`` 只在 ``oracle`` 臂作
# **上界诊断**用,不构成任何 reward。

**用法**(下面三条命令都以 ``PYTHONPATH=rl/code:code python3 <本文件>`` 开头)。
**冻结守卫**:容器里必须 ``transformers==5.6.0`` —— 版本不符时
``GUIOwlOSWorldRuntime`` 的 snapshot 校验会直接拒绝启动(踩过的坑,不要
用 pip 里默认的新版本)。同时必须提供 ``--snapshot-manifest``。

    # GPU 真跑(容器内,8 shard 之一)
    --model-dir /data/models/GUI-Owl-1.5-8B \\
    --snapshot-manifest /data/models/GUI-Owl-1.5-8B.manifest.json \\
    --out-dir /data/agentic_sensitivity/run01 --split syn_iid --n-tasks 200 \\
    --device cuda:0 --shard-index 0 --shard-count 8

    # 无 GPU 的管线自检(macOS 开发机即可)
    --dry-run --limit-tasks 3 --out-dir /tmp/sens_dry

    # 归约:多个 shard 的 JSONL 合成一份汇总
    --reduce /data/agentic_sensitivity/run01/records.shard*.jsonl \\
    --summary /data/agentic_sensitivity/run01/summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]
for _extra in (_ROOT / "rl" / "code", _ROOT / "code", _HERE.parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from causalcache_agentic.contract import StepRecord, TaskSpec, Trajectory
from causalcache_agentic.env import GUIEnv, action_line, normalize_action
from causalcache_agentic.expert import oracle_core_steps, subset_is_solvable
from causalcache_agentic.policy_io import HistoryFrameBank, PolicyInputBuilder

ARMS = ("recent2", "oracle", "random2", "none")
DEFAULT_BUDGET = 2
FALLBACK_ACTION = {"action": "wait"}


# ---------------------------------------------------------------- 确定性工具

def stable_seed(*parts: Any) -> int:
    """由若干标识拼出可复现的整数 seed(跨进程稳定,不依赖 PYTHONHASHSEED)。

    # note (luojiaxuan): 用 sha256 而非内置 hash —— 后者带进程盐,同一命令两次跑
    # 会给出不同的 random2 子集,直接毁掉"确定性"这条硬约束。
    """
    raw = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def read_probe(task: TaskSpec) -> tuple[tuple[int, ...], int | None]:
    """读 ``memory_probe``:(required_steps 升序, decision_step)。**纯诊断读口** ——
    前者只喂 ``oracle`` 臂作上界对照,后者只进记录与 dry-run 假策略,均非 reward。
    """
    probe = task.memory_probe or {}
    raw = probe.get("required_steps")
    req = (tuple(sorted({int(v) for v in raw if isinstance(v, (int, float))}))
           if isinstance(raw, (list, tuple)) else ())
    dec = probe.get("decision_step")
    return req, (int(dec) if isinstance(dec, (int, float)) else None)


# ---------------------------------------------------------------- 记忆臂

@dataclass(frozen=True)
class SubsetChoice:
    """一步的子集决策结果 + 诊断(诊断字段禁止回流到 policy 输入)。

    ``covered``:该步所需**变量**是否已被"被选帧 + 当前帧"覆盖,口径来自
    ``expert.subset_is_solvable`` —— 不是"required_steps ⊆ subset"的帧号包含关系
    (required_steps 含大量冗余帧,按帧号比会把可解的子集误判成没覆盖)。
    """

    subset: tuple[int, ...]
    logprob: float | None = None
    required_available: tuple[int, ...] = ()
    required_missing: tuple[int, ...] = ()
    is_decision_step: bool = False
    covered: bool = True


def arm_budget(arm: str, budget: int) -> int:
    """各臂的名义预算 B(``none`` 臂 B=0,其余用全局 ``--budget``)。"""
    return 0 if arm == "none" else budget


def select_subset(arm: str, bank: HistoryFrameBank, step: int, task: TaskSpec,
                  budget: int, seed: int) -> SubsetChoice:
    """按臂规则挑历史帧。**确定性**:同 (arm, task, step, seed) 恒给同一子集。

    候选口径完全交给 ``HistoryFrameBank.candidates``:合法事件号 ∈ [1, step-1]
    ——帧 0 恒被折叠进首轮文本,当前帧 step 恒在末轮,两者都不可再选。
    """
    cands = bank.candidates(step)
    want = min(arm_budget(arm, budget), len(cands))
    all_required, decision = read_probe(task)
    required = tuple(j for j in all_required if j < step)
    pool = set(cands)
    avail = tuple(j for j in required if j in pool)
    missing = tuple(j for j in required if j not in pool)
    at_decision = decision == step

    def _out(subset: tuple[int, ...], logp: float | None = None) -> SubsetChoice:
        return SubsetChoice(subset, logp, avail, missing, at_decision,
                            subset_is_solvable(task, step, subset))

    if want == 0:
        return _out(())
    if arm == "recent2":
        return _out(tuple(cands[-want:]))
    if arm == "random2":
        rng = random.Random(stable_seed(seed, task.task_id, "random2", step))
        return _out(tuple(sorted(rng.sample(cands, want))),
                    -math.log(math.comb(len(cands), want)))
    if arm == "oracle":
        # note (luojiaxuan): 必需帧的挑法直接调 expert.oracle_core_steps(变量级覆盖,
        # 由近及远贪心),**不另立新口径**;它给不满预算时才用 recent 补位。
        core = [j for j in oracle_core_steps(task, step, want) if j in pool]
        pad = [j for j in reversed(cands) if j not in core][: want - len(core)]
        return _out(tuple(sorted(core + pad)))
    raise ValueError(f"未知 arm {arm!r},合法值 {ARMS}")


# ---------------------------------------------------------------- 策略后端

class DryRunPolicy:
    """无 GPU 的假策略,只用于验证 rollout / prompt / 统计管线是否跑得通。

    两种模式(``--dry-run-mode``):
      * ``informed``(默认)—— 非 memory 关键步照抄专家动作;在
        ``memory_probe.decision_step`` 这一步按"必需帧是否都在被选子集里"投一枚
        **确定性硬币**决定照抄还是乱点。四条臂因此给出不同成功率,
        McNemar 与分层统计路径才真正被走到;
      * ``random`` —— 每步都乱点,纯压力测试(所有臂近似全败)。
    这里的成功率**没有任何科学含义**:它读了专家动作与 memory_probe,
    只用来确认脚本本身没坏。真跑一律走 ``FrozenGUIOwlPolicy``。

    # note (luojiaxuan): 硬币 seed 取自 (task, step, **被选子集**) 而非臂名 —— 冻结
    # 策略是 greedy 的,两臂选到同一子集就必须给出同一条轨迹;假策略照此建模,自检
    # 才抓得到"臂身份泄漏进 prompt / 记账串台"这类 bug。
    """

    def __init__(self, mode: str, seed: int, p_covered: float = 0.95,
                 p_blind: float = 0.10) -> None:
        self.mode, self.seed = mode, seed
        self.p_covered, self.p_blind = p_covered, p_blind
        self.metadata = {"policy_profile_id": "dry_run_fake", "mode": mode,
                         "frozen": False, "p_covered": p_covered, "p_blind": p_blind}

    def generate(self, messages: list[dict[str, Any]], ctx: dict[str, Any]) -> str:
        task: TaskSpec = ctx["task"]
        step: int = ctx["step"]
        choice: SubsetChoice = ctx["choice"]
        rng = random.Random(
            stable_seed(self.seed, task.task_id, step, choice.subset))
        expert = (dict(task.expert_actions[step])
                  if step < len(task.expert_actions) else dict(FALLBACK_ACTION))
        if self.mode == "random":
            act: dict[str, Any] = {"action": "left_click", "coordinate": [
                rng.randrange(1000), rng.randrange(1000)]}
        else:
            gated = choice.is_decision_step and bool(choice.required_available)
            p = self.p_covered if (not gated or choice.covered) else self.p_blind
            act = expert if rng.random() < p else {"action": "left_click",
                                                   "coordinate": [3, 3]}
        return "<tool_call>" + json.dumps(
            {"name": "computer_use", "arguments": act}) + "</tool_call>"


class FrozenGUIOwlPolicy:
    """冻结 GUI-Owl-1.5-8B 运行时(解码参数逐行镜像 rl_score_chosen_subsets.py)。"""

    def __init__(self, model_dir: Path, snapshot_manifest: Path, device: str,
                 visual_tokens: int, max_new_tokens: int,
                 adapter: Path | None = None) -> None:
        import torch

        from causalcache.osworld_gui_owl import _TOOL_SPEC, GUIOwlOSWorldRuntime

        self._torch, self._tool_spec = torch, _TOOL_SPEC
        self.max_new_tokens = max_new_tokens
        self.runtime = GUIOwlOSWorldRuntime(
            model_dir=model_dir, expected_snapshot_manifest=snapshot_manifest,
            device=device, effective_visual_tokens_per_image=visual_tokens,
            max_new_tokens=max_new_tokens)
        self.metadata = dict(self.runtime.metadata)
        # note (luojiaxuan): Phase 2 闭环评测要用 sparse-history SFT 出来的
        # adapter(policy_mem_sft)。冻结守卫只管基座权重,LoRA 是显式外挂,
        # 加载与否写进 metadata,避免"评的是哪个策略"这种记账含糊。
        self.adapter = None
        if adapter is not None:
            import sys as _sys
            for cand in (Path(__file__).resolve().parents[3] / "code" / "scripts",
                         Path("/data/osworld/CausalCache/code/scripts")):
                if cand.exists():
                    _sys.path.insert(0, str(cand))
                    break
            from train_success_sft_lora import inject_lora, load_lora_state_dict
            bundle = torch.load(adapter, map_location="cpu", weights_only=False)
            wrapped = inject_lora(
                self.runtime.model, rank=int(bundle["rank"]),
                alpha=int(bundle["alpha"]),
                target_modules=tuple(str(bundle.get(
                    "target_modules", "q_proj,k_proj,v_proj,o_proj")).split(",")),
                torch=torch,
                last_layer_count=(int(bundle["last_layers"])
                                  if bundle.get("last_layers") else None))
            load_lora_state_dict(wrapped, bundle["state"])
            self.adapter = {"path": str(adapter), "arm": bundle.get("arm"),
                            "epoch": bundle.get("epoch"),
                            "modules": len(wrapped)}
            self.metadata["adapter"] = self.adapter
            print(json.dumps({"adapter_loaded": self.adapter},
                             ensure_ascii=False), flush=True)

    def generate(self, messages: list[dict[str, Any]], ctx: dict[str, Any]) -> str:
        runtime = self.runtime
        enc = runtime.processor.apply_chat_template(
            messages, tools=[self._tool_spec], tokenize=True,
            add_generation_prompt=True, return_dict=True,
            return_tensors="pt").to(runtime.device)
        prompt_tokens = int(enc["input_ids"].shape[1])
        with self._torch.inference_mode():
            gt = runtime.generation_tokens
            out = runtime.model.generate(
                **enc, do_sample=False, max_new_tokens=self.max_new_tokens,
                eos_token_id=gt.tool_call_close_token_id,
                pad_token_id=gt.pad_token_id,
                suppress_tokens=list(gt.standard_eos_token_ids),
                num_beams=1, num_return_sequences=1)
        return runtime.processor.batch_decode(
            out[:, prompt_tokens:], skip_special_tokens=False,
            clean_up_tokenization_spaces=False)[0]


def _load_parse_tool_call() -> Callable[[str], dict[str, Any] | None]:
    """复用同目录 rl_oracle_enumerate 的解析器(与冻结评测逐字同源)。"""
    from rl_oracle_enumerate import parse_tool_call
    return parse_tool_call


# ---------------------------------------------------------------- 单条 rollout

def _shot_dir_for(shot_root: Path, task_id: str, arm: str) -> Path:
    safe = task_id.replace("::", "__").replace("/", "_")
    return shot_root / safe / arm


def run_episode(task: TaskSpec, arm: str, policy: Any, env: GUIEnv,
                parse: Callable[[str], dict[str, Any] | None], shot_root: Path,
                budget: int, seed: int, max_steps: int | None,
                keep_shots: bool) -> dict[str, Any]:
    """闭环跑一条轨迹:渲染当前帧 → 选 subset → 组 prompt → 生成 → 解析 → step。

    终止条件:policy 自己 terminate,或达到 ``task.max_steps``(``--max-steps`` 可
    再收紧)。成功与否**只**由 ``env.verify()`` 的声明式断言给出。返回一条可直接
    落 JSONL 的记录(内含完整 ``Trajectory``)。
    """
    builder = PolicyInputBuilder(budget=arm_budget(arm, budget))
    bank = HistoryFrameBank()
    shot_dir = _shot_dir_for(shot_root, task.task_id, arm)
    shot_dir.mkdir(parents=True, exist_ok=True)

    env.reset(task)
    history_actions: list[dict[str, Any]] = []
    records: list[StepRecord] = []
    limit = task.max_steps if max_steps is None else min(task.max_steps, max_steps)
    parse_failures = step_errors = 0
    decision_covered: bool | None = None
    probe_required, probe_decision = read_probe(task)
    started = time.perf_counter()

    for k in range(limit):
        shot = str(shot_dir / f"step{k:02d}.png")
        env.render(shot)
        bank.add(k, shot)
        choice = select_subset(arm, bank, k, task, budget, seed)
        messages = builder.build(task.instruction, history_actions,
                                 choice.subset, bank, k)
        ctx = {"task": task, "arm": arm, "step": k, "choice": choice}
        raw = policy.generate(messages, ctx)
        args = parse(raw)
        if not isinstance(args, dict) or "action" not in args:
            parse_failures += 1
            args = dict(FALLBACK_ACTION)
        else:
            args = normalize_action(args)
        try:
            line = action_line(args)
        except (KeyError, TypeError, ValueError):
            # note (luojiaxuan): 模型给出无法渲染成官方历史行的动作时不能中断
            # episode —— 记一条可读占位,让后续轮次的历史文本仍然合法。
            line = f"Action: {json.dumps(args, ensure_ascii=False)}"
        _obs, done, info = env.step(args)
        if info["last_error"]:
            step_errors += 1
        records.append(StepRecord(
            step=k, screenshot=shot, action=dict(args), action_line=line,
            source="policy", shown_subset=tuple(choice.subset),
            selector_logprob=choice.logprob,
            info={"arm": arm, "n_candidates": len(bank.candidates(k)),
                  "active_app": info["active_app"],
                  "last_error": info["last_error"],
                  "required_available": list(choice.required_available),
                  "required_unreachable": list(choice.required_missing),
                  "is_decision_step": choice.is_decision_step,
                  "required_covered": choice.covered}))
        if choice.is_decision_step:
            decision_covered = choice.covered
        history_actions.append(dict(args))
        if done:
            break

    elapsed = time.perf_counter() - started
    success = env.verify()
    traj = Trajectory(
        task_id=task.task_id, template_id=task.template_id, family=task.family,
        regime=task.regime, seed=task.seed, steps=records, success=success,
        reason=env.reason or "actions_exhausted", final_flags=dict(env.state.flags))
    record = {
        "task_id": task.task_id, "template_id": task.template_id,
        "family": task.family, "regime": task.regime, "seed": task.seed,
        "arm": arm, "budget": arm_budget(arm, budget),
        "success": bool(success), "reason": traj.reason,
        "n_steps": len(records), "max_steps": task.max_steps,
        "parse_failures": parse_failures, "step_errors": step_errors,
        "elapsed_sec": round(elapsed, 3), "shot_dir": str(shot_dir),
        "subsets": [list(r.shown_subset) for r in records],
        "final_flags": traj.final_flags,
        # note (luojiaxuan): 诊断专用 —— 禁止进 policy 输入,禁止作 RL reward。
        "diagnostics": {"required_steps": list(probe_required),
                        "decision_step": probe_decision,
                        "decision_step_covered": decision_covered},
        "trajectory": json.loads(traj.to_json())}
    if not keep_shots:
        for png in shot_dir.glob("*.png"):
            png.unlink(missing_ok=True)
    return record


# ---------------------------------------------------------------- 统计

def mcnemar_exact_p(b: int, c: int) -> float:
    """配对二值的 McNemar **精确**检验(双侧);b/c 为两类不一致对的计数。"""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def _rate(hits: int, n: int) -> float | None:
    return None if n == 0 else hits / n


def _arm_stats(rows: Sequence[dict[str, Any]], arms: Sequence[str]
               ) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for arm in arms:
        sel = [r for r in rows if r["arm"] == arm]
        hits = sum(1 for r in sel if r["success"])
        out[arm] = {
            "n": len(sel), "success": hits, "rate": _rate(hits, len(sel)),
            "mean_steps": (round(sum(r["n_steps"] for r in sel) / len(sel), 2)
                           if sel else None),
            "parse_failures": sum(r["parse_failures"] for r in sel)}
    return out


def _paired(rows: Sequence[dict[str, Any]], arm_a: str, arm_b: str
            ) -> dict[str, Any]:
    """arm_a 相对 arm_b 的配对差:只统计**两臂都跑过**的 task。"""
    by_task: dict[str, dict[str, bool]] = {}
    for r in rows:
        if r["arm"] in (arm_a, arm_b):
            by_task.setdefault(r["task_id"], {})[r["arm"]] = bool(r["success"])
    both = [v for v in by_task.values() if arm_a in v and arm_b in v]
    n = len(both)
    win = sum(1 for v in both if v[arm_a] and not v[arm_b])
    loss = sum(1 for v in both if v[arm_b] and not v[arm_a])
    agree_hit = sum(1 for v in both if v[arm_a] and v[arm_b])
    return {"arm_a": arm_a, "arm_b": arm_b, "n_paired": n,
            "a_rate": _rate(sum(1 for v in both if v[arm_a]), n),
            "b_rate": _rate(sum(1 for v in both if v[arm_b]), n),
            "delta": (None if n == 0 else (win - loss) / n),
            "rescue": win, "regression": loss, "both_success": agree_hit,
            "both_fail": n - win - loss - agree_hit,
            "mcnemar_p": mcnemar_exact_p(win, loss)}


def summarize(rows: Sequence[dict[str, Any]], arms: Sequence[str],
              reference: str, alpha: float, meta: dict[str, Any]
              ) -> dict[str, Any]:
    """汇总:整体成功率 + 对参照臂的配对差 + McNemar p + 分 regime / template。"""
    present = [a for a in arms if any(r["arm"] == a for r in rows)]
    pairs = [a for a in present if a != reference]
    overall_pairs = {f"{a}_vs_{reference}": _paired(rows, a, reference)
                     for a in pairs}

    def _strata(key: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for lv in sorted({r[key] for r in rows}):
            sub = [r for r in rows if r[key] == lv]
            out[str(lv)] = {"per_arm": _arm_stats(sub, present),
                            "paired": {f"{a}_vs_{reference}":
                                       _paired(sub, a, reference) for a in pairs}}
        return out

    sig = {k: v for k, v in overall_pairs.items()
           if v["mcnemar_p"] is not None and v["mcnemar_p"] < alpha}
    deltas = [abs(v["delta"]) for v in overall_pairs.values()
              if v["delta"] is not None]
    return {
        "meta": meta, "n_records": len(rows), "arms": present,
        "n_tasks": len({r["task_id"] for r in rows}),
        "reference_arm": reference,
        "per_arm": _arm_stats(rows, present),
        "paired_vs_reference": overall_pairs,
        "by_regime": _strata("regime"),
        "by_template": _strata("template_id"),
        "acceptance": {
            "criterion": (f"任一臂对 {reference} 的 McNemar 精确检验 p < {alpha}"
                          " ⇒ memory subset 改变了任务成功概率"),
            "alpha": alpha, "significant_pairs": sorted(sig),
            "max_abs_paired_delta": max(deltas) if deltas else None,
            "subset_changes_success": bool(sig)},
    }


# ---------------------------------------------------------------- IO

def read_records(paths: Sequence[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    """读若干 JSONL,按 (task_id, arm) 去重(后写覆盖先写);缺文件当空处理。

    # note (luojiaxuan): 断点续跑与多 shard 归约共用这一个读取口径,避免两处各写
    # 一份解析逻辑后续跑判重与最终统计对不上账。
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "task_id" in d and "arm" in d:
                    merged[(str(d["task_id"]), str(d["arm"]))] = d
    return merged


def build_tasks(args: argparse.Namespace) -> list[TaskSpec]:
    """按 ``--split`` 或 ``--families`` 实例化任务(随机只发生在这一步,带显式 seed)。"""
    from causalcache_agentic import tasks as tasks_module
    if args.families:
        fams = [f for item in args.families for f in item.split(",") if f]
        out = tasks_module.generate_batch(args.n_tasks, args.task_seed,
                                          families=fams)
    else:
        out = tasks_module.make_dataset(args.split, args.n_tasks, args.task_seed)
    if args.limit_tasks:
        out = out[: args.limit_tasks]
    return out


# ---------------------------------------------------------------- CLI

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="agentic_memory_sensitivity",
        description="不同 memory subset 是否改变任务成功概率(Phase 1 验收)")
    p.add_argument("--out-dir", type=Path, default=Path("/data/agentic_sensitivity"),
                   help="输出根目录(Linux 训练机默认落 /data 下)")
    p.add_argument("--output", type=Path, default=None,
                   help="逐 task 逐臂 JSONL(默认 <out-dir>/records.jsonl)")
    p.add_argument("--summary", type=Path, default=None,
                   help="汇总 JSON(默认 <out-dir>/summary.json)")
    p.add_argument("--shot-dir", type=Path, default=None,
                   help="截图根目录(默认 <out-dir>/shots)")
    p.add_argument("--reduce", type=Path, nargs="+", default=None,
                   help="只归约:读入这些 JSONL 直接出汇总,不跑任何 rollout")
    p.add_argument("--split", default="syn_iid",
                   choices=("train", "syn_iid", "syn_ood"))
    p.add_argument("--families", nargs="+", default=None,
                   help="按 family/template_id 过滤(逗号或空格分隔),给了就忽略 --split")
    p.add_argument("--n-tasks", type=int, default=100)
    p.add_argument("--task-seed", type=int, default=0)
    p.add_argument("--limit-tasks", type=int, default=0, help="0 = 不限")
    p.add_argument("--arms", nargs="+", default=list(ARMS),
                   help=f"逗号或空格分隔,合法值 {ARMS}")
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="B")
    p.add_argument("--reference-arm", default="recent2",
                   help="配对差与 McNemar 的参照臂")
    p.add_argument("--arm-seed", type=int, default=0, help="random2 臂的采样 seed")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=None,
                   help="额外的步数上限(与 task.max_steps 取小)")
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--adapter", type=Path, default=None,
                   help="sparse-history SFT 的 LoRA adapter(Phase 2 闭环评测);"
                        "不给 = 冻结基线策略")
    p.add_argument("--snapshot-manifest", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--dry-run", action="store_true",
                   help="不加载模型,用假策略验证管线(无需 GPU)")
    p.add_argument("--dry-run-mode", default="informed",
                   choices=("informed", "random"))
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--cleanup-shots", action="store_true",
                   help="每条 episode 跑完即删其 PNG(省磁盘,牺牲可复查性)")
    p.add_argument("--progress-every", type=int, default=10)
    return p.parse_args(list(argv) if argv is not None else None)


def _resolve_paths(args: argparse.Namespace) -> None:
    args.out_dir = args.out_dir.expanduser()
    args.output = args.output or args.out_dir / "records.jsonl"
    args.summary = args.summary or args.out_dir / "summary.json"
    args.shot_dir = args.shot_dir or args.out_dir / "shots"
    if args.shard_count > 1:
        args.output = args.output.with_name(
            f"{args.output.stem}.shard{args.shard_index:02d}{args.output.suffix}")
        args.summary = args.summary.with_name(
            f"{args.summary.stem}.shard{args.shard_index:02d}{args.summary.suffix}")


def _emit_summary(args: argparse.Namespace, rows: Sequence[dict[str, Any]],
                  arms: Sequence[str], meta: dict[str, Any]) -> dict[str, Any]:
    summary = summarize(rows, arms, args.reference_arm, args.alpha, meta)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(json.dumps(summary["per_arm"], ensure_ascii=False))
    print(json.dumps(summary["acceptance"], ensure_ascii=False))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _resolve_paths(args)
    arms = [a for item in args.arms for a in str(item).split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"未知 arm {bad},合法值 {list(ARMS)}")

    if args.reduce:
        _emit_summary(args, list(read_records(list(args.reduce)).values()), arms,
                      {"mode": "reduce",
                       "sources": [str(p) for p in args.reduce]})
        return 0

    if not args.dry_run and (args.model_dir is None or
                             args.snapshot_manifest is None):
        raise SystemExit("真跑必须给 --model-dir 与 --snapshot-manifest"
                         "(冻结守卫);只想验管线请加 --dry-run")

    shard = [t for i, t in enumerate(build_tasks(args))
             if i % args.shard_count == args.shard_index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.shot_dir.mkdir(parents=True, exist_ok=True)
    done = set(read_records([args.output]))

    parse = _load_parse_tool_call()
    policy: Any = (DryRunPolicy(args.dry_run_mode, args.arm_seed) if args.dry_run
                   else FrozenGUIOwlPolicy(args.model_dir, args.snapshot_manifest,
                                           args.device, args.visual_tokens,
                                           args.max_new_tokens,
                                           adapter=args.adapter))
    env = GUIEnv(auto_stop_on_success=False)
    n_new = 0
    n_err = 0
    started = time.perf_counter()
    with args.output.open("a", encoding="utf-8") as sink:
        for task in shard:
            for arm in arms:
                if (task.task_id, arm) in done:
                    continue
                # note (luojiaxuan): 逐 (task, arm) 兜底 —— 策略输出不可控,
                # 一个任务抛异常不能打崩整个分片(2026-08-13 实测:策略偶发
                # action="click" 让 2/4 分片直接退出,已在 policy_io 归一别名,
                # 这里再加一层"记账为失败并继续"的护栏)。失败按 terminal
                # reward 的定义就是 success=False,不额外惩罚也不静默丢弃。
                try:
                    rec = run_episode(task, arm, policy, env, parse,
                                      args.shot_dir, args.budget, args.arm_seed,
                                      args.max_steps,
                                      keep_shots=not args.cleanup_shots)
                except Exception as exc:  # noqa: BLE001
                    rec = {"task_id": task.task_id,
                           "template_id": task.template_id,
                           "family": task.family, "regime": task.regime,
                           "seed": task.seed, "arm": arm, "success": False,
                           "reason": "episode_error", "n_steps": 0,
                           "error": f"{type(exc).__name__}: {exc}"[:400]}
                    n_err += 1
                    print(json.dumps({"shard": args.shard_index,
                                      "episode_error": rec["error"],
                                      "task_id": task.task_id, "arm": arm},
                                     ensure_ascii=False), flush=True)
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sink.flush()
                n_new += 1
                if args.progress_every and n_new % args.progress_every == 0:
                    print(json.dumps({"shard": args.shard_index, "done": n_new,
                                      "elapsed_sec": round(
                                          time.perf_counter() - started, 1)},
                                     ensure_ascii=False), flush=True)

    meta = {"mode": "dry_run" if args.dry_run else "frozen_gui_owl",
            "policy": getattr(policy, "metadata", {}),
            "budget": args.budget, "arm_seed": args.arm_seed,
            "split": args.split, "families": args.families,
            "n_tasks_requested": args.n_tasks, "task_seed": args.task_seed,
            "shard_index": args.shard_index, "shard_count": args.shard_count,
            "output": str(args.output), "shot_dir": str(args.shot_dir),
            "new_records": n_new, "episode_errors": n_err, "wall_sec": round(time.perf_counter() - started, 1),
            "reward": "terminal_verifier_only",
            "note": "memory_probe 仅用于 oracle 臂的上界诊断,不作 RL reward"}
    print(json.dumps({"records": str(args.output), "summary": str(args.summary),
                      "new_records": n_new}, ensure_ascii=False))
    _emit_summary(args, list(read_records([args.output]).values()), arms, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
