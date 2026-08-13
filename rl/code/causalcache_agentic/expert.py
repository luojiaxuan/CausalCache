"""专家轨迹自检与 Phase 2 SFT 记录生成。

# note (luojiaxuan): 两件事,一件都不多做:
#   1. ``expert_success_rate`` —— 无渲染回放全部任务,给出总体/分层成功率与
#      **足够定位的失败样例**(第一条不满足的断言 + 实际值 + flags 轨迹),
#      用来区分「tasks.py 造错了任务」与「executor.py 执行错了动作」;
#   2. ``build_sft_records`` —— 把专家轨迹转成 Phase 2 gated sparse-history SFT
#      记录,并按路线图硬约束标注 ``solvable``:**禁止产出"输入不含答案却
#      要求 gold"的样本**。
# memory_probe / env_params 一律只进 ``diagnostics`` 与元数据字段,
# **不得进 policy 输入,也不得作 RL reward**(contract.py 与路线图 §2)。
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Sequence

from .contract import (
    NORM_MAX,
    SCREEN_H,
    SCREEN_W,
    Assertion,
    ScreenState,
    TaskSpec,
    Trajectory,
    verify as verify_assertions,
)
from .env import GUIEnv, replay_detailed, rollout_expert

# note (luojiaxuan): 这三种 regime 的关键证据不在 recent-2 里(见路线图 §2)。
HISTORY_DEMANDING_REGIMES = ("one_old_frame", "two_frame_complementary",
                             "distractor_heavy")
SUBSET_POLICIES = ("recent2", "oracle", "oracle_plus_distractor")
DEFAULT_BUDGET = 2


# ---------------------------------------------------------------- 断言诊断

def _assertion_actual(state: ScreenState, a: Assertion) -> Any:
    """取该断言实际观测到的值(找不到 widget 返回 None,便于区分'没写'与'写错')。"""
    if a.kind in ("flag_equals", "flag_contains", "flag_absent"):
        return state.flags.get(a.target)
    for win in state.windows.values():
        for w in win.widgets:
            if w.wid == a.target:
                return w.value
    return None


def _failed_assertions(state: ScreenState, assertions: Sequence[Assertion]
                       ) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, a in enumerate(assertions):
        if verify_assertions(state, [a]):
            continue
        out.append({
            "index": i, "kind": a.kind, "target": a.target,
            "expected": a.expected, "actual": _assertion_actual(state, a),
        })
    return out


def _hint(task: TaskSpec, res: Any, failed: list[dict[str, Any]]) -> str:
    """把失败归因到最可能的模块,减少来回排查。"""
    if res.last_error:
        return ("executor 在执行中抛异常(见 last_error 与 trace.error):优先查 "
                "executor.py 的动作分支或 effect 解释")
    if res.reason == "max_steps":
        return ("跑满 max_steps 仍未终止:多半是 tasks.py 的 expert_actions 过长或 "
                "max_steps 过小")
    if res.steps_executed < len(task.expert_actions):
        return ("专家动作未跑完就终止(提前 terminate / 环境判死):查 tasks.py 的 "
                "expert_actions 顺序")
    if not failed:
        return "断言全过但被判失败——不应发生,查 verify 调用路径"
    first = failed[0]
    if first["actual"] is None:
        if first["kind"].startswith("flag"):
            return (f"flag {first['target']!r} 从未被写入:多半是某次点击没命中控件,"
                    "或该控件缺少 set_flag/flag_from effect(tasks.py 或 apps.py)")
        return (f"断言目标 widget {first['target']!r} 在终局 state 里不存在:"
                "查 tasks.py 的 widget id 或该窗口是否被关闭")
    if first["actual"] == "":
        return (f"{first['target']!r} 存在但值为空:点击命中了但 effect 没带值,"
                "或 type 动作没写进 focus 控件(查 executor.py 的 focus/type 语义)")
    return (f"{first['target']!r} 实际值 {first['actual']!r} != 期望 "
            f"{first['expected']!r}:值算错,多半是 tasks.py 的期望值构造")


# ---------------------------------------------------------------- 成功率自检

def expert_success_rate(tasks: list[TaskSpec], *, executor: Any = None,
                        max_failures: int = 20) -> dict[str, Any]:
    """对每个 task 跑无渲染 replay,返回总体与分层成功率 + 失败样例。

    验收口径见路线图 §2「Phase 1 验收」:scripted expert 成功率应≈100%。
    """
    n_success = 0
    by_template: dict[str, dict[str, Any]] = {}
    by_regime: dict[str, dict[str, Any]] = {}
    by_family: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}

    for task in tasks:
        res = replay_detailed(task, list(task.expert_actions), executor=executor,
                              with_trace=False)
        ok = res.success
        n_success += int(ok)
        reasons[res.reason] = reasons.get(res.reason, 0) + 1
        for bucket, key in ((by_template, task.template_id),
                            (by_regime, task.regime), (by_family, task.family)):
            row = bucket.setdefault(key, {"n": 0, "n_success": 0})
            row["n"] += 1
            row["n_success"] += int(ok)
        if not ok and len(failures) < max_failures:
            detail = replay_detailed(task, list(task.expert_actions),
                                     executor=executor, with_trace=True)
            failed = _failed_assertions(detail.state, task.assertions)
            failures.append({
                "task_id": task.task_id, "template_id": task.template_id,
                "family": task.family, "regime": task.regime, "seed": task.seed,
                "instruction": task.instruction,
                "reason": detail.reason, "last_error": detail.last_error,
                "steps_executed": detail.steps_executed,
                "n_expert_actions": len(task.expert_actions),
                "max_steps": task.max_steps,
                "first_failed_assertion": failed[0] if failed else None,
                "failed_assertions": failed,
                "final_flags": dict(detail.state.flags),
                "active_app": detail.state.active_app,
                "focus": list(detail.state.focus) if detail.state.focus else None,
                "terminated": detail.state.terminated,
                "terminate_status": detail.state.terminate_status,
                "clipboard": detail.state.clipboard,
                "trace": detail.trace,
                "hint": _hint(task, detail, failed),
            })

    for bucket in (by_template, by_regime, by_family):
        for row in bucket.values():
            row["success_rate"] = row["n_success"] / max(row["n"], 1)
    n = len(tasks)
    return {
        "n_tasks": n, "n_success": n_success,
        "success_rate": n_success / max(n, 1),
        "by_template": by_template, "by_regime": by_regime, "by_family": by_family,
        "reasons": reasons,
        "n_failures_total": n - n_success,
        "n_failures_reported": len(failures),
        "failures": failures,
    }


# ---------------------------------------------------------------- memory probe

def _as_int_list(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value if isinstance(v, (int, float, str))
                     and str(v).lstrip("-").isdigit())
    return ()


def _probe_steps(task: TaskSpec, key: str, step: int) -> tuple[int, ...] | None:
    """读取 memory_probe 里的步号标注,并裁到"该决策步之前**且合法**"的帧。

    支持全局 list 与 {decision_step: [...]} 两种写法。返回 None 表示**未标注**
    (与"标注为空"区分开:后者意味着该步不需要历史)。
    合法区间与 ``policy_io.HistoryFrameBank.candidates`` 一致:``[1, step-1]``
    —— 帧 0 恒被折叠成首轮文本,不能作历史保留图。
    """
    probe = task.memory_probe or {}
    raw = probe.get(key)
    if raw is None:
        raw = probe.get(f"{key}_by_step")
    if raw is None:
        return None
    if isinstance(raw, dict):
        for cand in (step, str(step)):
            if cand in raw:
                return _legal(_as_int_list(raw[cand]), step)
        return ()
    return _legal(_as_int_list(raw), step)


def _probe_all(task: TaskSpec, key: str) -> tuple[int, ...]:
    """同 ``_probe_steps`` 但**不按决策步裁剪**(算"这个任务总共需要哪些变量"时用)。"""
    raw = (task.memory_probe or {}).get(key)
    if isinstance(raw, dict):
        return tuple(sorted({s for v in raw.values() for s in _as_int_list(v)}))
    return tuple(sorted(set(_as_int_list(raw))))


def _legal(steps: Iterable[int], now: int) -> tuple[int, ...]:
    """裁到合法候选区间 [1, now-1] 并去重升序。"""
    return tuple(sorted({s for s in steps if 1 <= s < now}))


def _frame_vars(task: TaskSpec) -> dict[int, frozenset[str]]:
    """``memory_probe.frame_vars``:帧号 → 该帧屏幕上能读到的变量名集合。"""
    out: dict[int, frozenset[str]] = {}
    for key, val in ((task.memory_probe or {}).get("frame_vars") or {}).items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        out[idx] = frozenset(str(v) for v in (val or ()))
    return out


def _vars_at(fv: dict[int, frozenset[str]], steps: Iterable[int]) -> frozenset[str]:
    """给定若干帧号,这些帧屏幕上能读到的变量并集。"""
    out: set[str] = set()
    for s in steps:
        out |= fv.get(int(s), frozenset())
    return frozenset(out)


def _decision_steps(task: TaskSpec) -> frozenset[int]:
    """**消费历史证据**的那些步(其余步的动作完全由当前屏决定)。

    读 ``memory_probe.decision_steps``(list)或 ``decision_step``(int);
    都没有时退化为最后一步。
    """
    probe = task.memory_probe or {}
    many = _as_int_list(probe.get("decision_steps"))
    if many:
        return frozenset(many)
    one = probe.get("decision_step")
    if isinstance(one, (int, float)):
        return frozenset({int(one)})
    return frozenset({max(0, len(task.expert_actions) - 1)})


def _recent(step: int, budget: int, exclude: Iterable[int] = ()) -> list[int]:
    """当前决策步之前、由近及远的合法历史帧步号(用于补位)。

    # note (luojiaxuan): 起点是 1 而不是 0 —— 与 policy_io 的候选口径同源,
    # 帧 0 恒被折叠成首轮 "Previous actions" 文本,选它组不出合法 prompt。
    """
    taken = set(exclude)
    out = [i for i in range(step - 1, 0, -1) if i not in taken]
    return out[:budget]


def _cover_subset(task: TaskSpec, step: int, budget: int) -> list[int]:
    """特权(oracle)子集:在预算内**尽量覆盖尚缺的变量**,由近及远贪心。

    # note (luojiaxuan): 不能简单取 required_steps 的最后 B 张 —— tasks.py 把
    # "凡是能看到 v1/v2 的帧"都记进 required_steps(含大量冗余帧),最后 B 张
    # 可能恰好漏掉互补 regime 里那张只带 v2 的老帧。故按 frame_vars 做变量级覆盖。
    """
    fv = _frame_vars(task)
    need = _vars_at(fv, _probe_all(task, "required_steps")) - fv.get(step, frozenset())
    chosen: list[int] = []
    covered: set[str] = set()
    for s in sorted(_probe_steps(task, "required_steps", step) or (), reverse=True):
        if len(chosen) >= budget or not (need - covered):
            break
        if (fv.get(s, frozenset()) & need) - covered:
            chosen.append(s)
            covered |= fv.get(s, frozenset())
    return chosen


def oracle_core_steps(task: TaskSpec, step: int, budget: int) -> list[int]:
    """**公开口径**:特权 oracle 子集里"必需帧"那一半(升序,可能少于 budget)。

    SFT 记录(``build_sft_records``)与 Phase 1 敏感性 harness 的 ``oracle`` 臂
    共用本函数,避免两处各写一套"哪些帧算必需"而悄悄分叉。
    """
    return sorted(_cover_subset(task, step, budget))


def subset_is_solvable(task: TaskSpec, step: int, shown: Sequence[int]) -> bool:
    """**公开口径**:给了这个 subset,该步的专家动作是否仍可从输入推出来。"""
    return _solvability(task, step, shown)[0]


def _pick_subset(task: TaskSpec, step: int, policy: str, budget: int
                 ) -> tuple[list[int], str, dict[str, Any]]:
    """按 subset_policy 选出该步喂给 policy 的历史帧步号(升序)。**确定性**。"""
    required = _probe_steps(task, "required_steps", step)
    distractors = _probe_steps(task, "distractor_steps", step)
    diag = {"required_steps": None if required is None else list(required),
            "distractor_steps": None if distractors is None else list(distractors)}

    if policy == "recent2":
        return sorted(_recent(step, budget)), "recent2", diag

    core = _cover_subset(task, step, budget)
    if policy == "oracle":
        chosen = core + _recent(step, budget - len(core), exclude=core)
        return sorted(chosen), "oracle", diag

    if policy == "oracle_plus_distractor":
        room = budget - len(core)
        if room <= 0:
            # note (luojiaxuan): 必需帧已占满预算,再塞干扰帧就会挤掉答案 ——
            # 那是被明令禁止的不可解样本,故退化为 oracle 并如实标注。
            return sorted(core), "oracle", diag
        extra = [s for s in (distractors or ()) if s not in core][:room]
        if len(extra) < room:
            extra += _recent(step, room - len(extra), exclude=core + extra)
        return sorted(core + extra), "oracle_plus_distractor", diag

    raise ValueError(f"未知 subset_policy {policy!r},合法值 {SUBSET_POLICIES}")


def _solvability(task: TaskSpec, step: int, shown: Sequence[int]
                 ) -> tuple[bool, str, str]:
    """判断"给了这个 subset,专家动作是否还可从输入推出来"。

    判据是**变量级覆盖**而不是"required_steps ⊆ shown_subset":后者含大量冗余帧
    (同一个变量往往在多帧重复出现),按帧号比会把可解样本误判成不可解。
    两条口径:
      * ``step`` 不是 decision step —— 该步动作完全由当前屏决定(切应用、点开页签、
        点 Save、terminate),不消费历史证据,恒可解;
      * 是 decision step —— 要求 ``需要的变量 ⊆ (被选帧 ∪ 当前帧) 上可读到的变量``。
    """
    fv = _frame_vars(task)
    required = _probe_all(task, "required_steps")
    decisions = _decision_steps(task)
    if not fv or not required:
        # note (luojiaxuan): 没有 frame_vars 标注时退回帧号口径,并对"需要历史"的
        # regime 保守判死 —— 这同时是给 tasks.py 的提醒:请补 memory_probe。
        legal = _probe_steps(task, "required_steps", step)
        if legal is not None:
            missing = [s for s in legal if s not in set(shown)]
            if missing:
                return False, f"required_steps {missing} 不在 shown_subset 中", "known"
            return True, "required_steps 全部在 shown_subset 中", "known"
        if task.regime in HISTORY_DEMANDING_REGIMES and step > 0:
            return (False,
                    f"memory_probe 未标注 required_steps,而 regime={task.regime} 需要"
                    "历史帧,保守判为不可解(请在 tasks.py 补 memory_probe)",
                    "conservative")
        return True, f"regime={task.regime} 不依赖历史帧", "known"

    if step not in decisions:
        return (True, f"step={step} 不在 decision step {sorted(decisions)} 里,"
                "该步动作由当前屏决定,不消费历史证据", "known")
    need = _vars_at(fv, required)
    got = _vars_at(fv, list(shown) + [step])
    missing_vars = sorted(need - got)
    if missing_vars:
        return (False, f"变量 {missing_vars} 在 shown_subset {list(shown)} 与当前帧 "
                f"{step} 上都读不到", "known")
    return True, f"所需变量 {sorted(need)} 已被 shown_subset + 当前帧覆盖", "known"


# ---------------------------------------------------------------- SFT 记录

def build_sft_records(traj: Trajectory, task: TaskSpec, subset_policy: str,
                      *, budget: int = DEFAULT_BUDGET) -> list[dict[str, Any]]:
    """把一条专家轨迹转成 Phase 2 SFT 记录(每个决策步一条)。

    subset_policy:
      * ``recent2`` —— 最近 B 帧,recent-B 基线与 "recent 足够" 类样本;
      * ``oracle`` —— 用 ``task.memory_probe["required_steps"]``;不足 B 张时
        用 recent 补齐,超过 B 张时取最近的 B 张;
      * ``oracle_plus_distractor`` —— 必需帧 + 干扰帧(distractor-robustness)。
        若必需帧已占满预算,**退化为 oracle**(``subset_policy_effective`` 会
        如实写 ``oracle``),因为挤掉必需帧就会造出不可解样本。

    **solvable 约定(路线图 §3 硬约束:禁止"输入不含答案却要求 gold"的样本)**:
      * ``memory_probe.required_steps`` 有标注时,``solvable = required ⊆ shown_subset``
        (只比较该决策步之前已存在的帧;更晚的帧不算欠账),``solvable_confidence="known"``;
      * 未标注且 ``regime`` 属于 {one_old_frame, two_frame_complementary,
        distractor_heavy} 时,保守判 ``solvable=False``、
        ``solvable_confidence="conservative"`` —— 这同时是给 tasks.py 的提醒:
        请补 memory_probe;
      * 其余(recent_sufficient / history_irrelevant)判 ``solvable=True``。
    **本函数不丢弃任何记录**:是否丢弃由调用方决定(SFT 训练应只用
    ``solvable=True`` 且 ``trajectory_success=True`` 的记录)。
    """
    if subset_policy not in SUBSET_POLICIES:
        raise ValueError(f"未知 subset_policy {subset_policy!r},合法值 {SUBSET_POLICIES}")
    shots = {s.step: s.screenshot for s in traj.steps}
    records: list[dict[str, Any]] = []
    for rec in traj.steps:
        k = rec.step
        prior = [s for s in traj.steps if s.step < k]
        shown, effective, diag = _pick_subset(task, k, subset_policy, budget)
        solvable, why, confidence = _solvability(task, k, shown)
        records.append({
            "record_id": f"{task.task_id}#{k:02d}",
            "task_id": task.task_id, "template_id": task.template_id,
            "family": task.family, "regime": task.regime, "seed": task.seed,
            "instruction": task.instruction,
            "step": k, "n_steps": len(traj.steps),
            "action_history": [{"step": s.step, "action_line": s.action_line}
                               for s in prior],
            "history_text": "\n".join(f"Step{s.step + 1}: {s.action_line}"
                                      for s in prior),
            "current_screenshot": rec.screenshot,
            "candidate_history": [{"step": s.step, "screenshot": s.screenshot,
                                   "age": k - s.step} for s in prior],
            "budget": budget,
            "subset_policy": subset_policy,
            "subset_policy_effective": effective,
            "shown_subset": shown,
            "shown_subset_screenshots": [shots[i] for i in shown if i in shots],
            "expert_action": dict(rec.action),
            "expert_action_line": rec.action_line,
            "expert_tool_call": {"name": "computer_use", "arguments": dict(rec.action)},
            "terminal_task_id": task.task_id,
            "env_params": dict(task.params),
            "max_steps": task.max_steps,
            "screen": {"w": SCREEN_W, "h": SCREEN_H, "norm_max": NORM_MAX},
            "trajectory_success": traj.success,
            "solvable": solvable,
            "solvable_reason": why,
            "solvable_confidence": confidence,
            # note (luojiaxuan): 诊断专用 —— 禁止进 policy 输入,禁止作 RL reward。
            "diagnostics": {**diag, "memory_probe": dict(task.memory_probe),
                            "note": "诊断字段,禁止进 policy 输入 / 禁止作 reward"},
        })
    return records


# ---------------------------------------------------------------- 脚本入口

def load_tasks(n_per_template: int = 4, seed: int = 0,
               split: str = "") -> list[TaskSpec]:
    """从同包 ``tasks`` 模块拿一批任务。

    ``split`` 为空 → 全部 template 轮转(``generate_batch`` 按 ``i % len(pool)``
    取模,故 n = n_per_template × template 数正好每个模板 n_per_template 个);
    给了 split(train / syn_iid / syn_ood)则走 ``make_dataset`` 的 template 级划分。

    # note (luojiaxuan): 定稿前这里是"多入口名 + 多签名"的探测胶水,而 tasks.py
    # 最终只暴露 generate_task / generate_batch / make_dataset —— 探测名一个都没
    # 命中,load_tasks 必抛 RuntimeError(expert 的 CLI 因此完全跑不起来)。
    # 现在直接绑定真实入口,签名漂移会在 import 期就暴露,而不是运行期才发现。
    """
    from . import tasks as tasks_module

    n_total = max(1, int(n_per_template)) * len(tasks_module.TEMPLATES)
    if split:
        return list(tasks_module.make_dataset(split, n_total, seed))
    return list(tasks_module.generate_batch(n=n_total, seed=seed))


def main(argv: Sequence[str] | None = None) -> int:
    """脚本入口:自检专家成功率,可选生成专家轨迹与 SFT 记录。返回 0 表示全通过。"""
    import argparse

    p = argparse.ArgumentParser(prog="causalcache_agentic.expert")
    p.add_argument("--n-per-template", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="", choices=("", "train", "syn_iid", "syn_ood"),
                   help="留空 = 全部 template;否则走 tasks.make_dataset 的 template 级划分")
    p.add_argument("--max-failures", type=int, default=20)
    p.add_argument("--subset-policy", default="recent2", choices=list(SUBSET_POLICIES))
    p.add_argument("--shot-dir", default="")
    p.add_argument("--sft-out", default="")
    p.add_argument("--traj-out", default="")
    p.add_argument("--only-solvable", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    tasks = load_tasks(args.n_per_template, args.seed, args.split)
    summary = expert_success_rate(tasks, max_failures=args.max_failures)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    if args.sft_out or args.traj_out:
        shot_dir = args.shot_dir or os.path.join(
            os.path.dirname(os.path.abspath(args.sft_out or args.traj_out)), "shots")
        env = GUIEnv(auto_stop_on_success=False)
        sft_f = open(args.sft_out, "w", encoding="utf-8") if args.sft_out else None
        traj_f = open(args.traj_out, "w", encoding="utf-8") if args.traj_out else None
        n_rec = 0
        try:
            for task in tasks:
                traj = rollout_expert(task, env, shot_dir)
                if traj_f is not None:
                    traj_f.write(traj.to_json() + "\n")
                if sft_f is None:
                    continue
                for rec in build_sft_records(traj, task, args.subset_policy):
                    if args.only_solvable and not (rec["solvable"]
                                                   and rec["trajectory_success"]):
                        continue
                    sft_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_rec += 1
        finally:
            for f in (sft_f, traj_f):
                if f is not None:
                    f.close()
        print(json.dumps({"sft_records": n_rec, "shot_dir": shot_dir},
                         ensure_ascii=False))
    return 0 if summary["success_rate"] >= 1.0 else 1


# note (luojiaxuan): 没有这段 guard,`python3 -m causalcache_agentic.expert` 只会
# 导入模块然后静默退出 0(踩过:日志空文件、产物一个没落,却看起来"跑成功了")。
if __name__ == "__main__":
    raise SystemExit(main())
