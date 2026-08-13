#!/usr/bin/env python3
"""E5 的**决策步诊断**:同一状态下换记忆子集,executor 的动作是否改变。

# note (luojiaxuan): 为什么需要这个脚本(2026-08-13 实测记录):
# 闭环四臂敏感性(agentic_memory_sensitivity.py)在**冻结** GUI-Owl 上撞到
# 地板效应 —— recent2/oracle/random2/none 的任务成功率分别 0.0/2.1/2.1/4.3%
# (n=189),且 161/189 条撞 max_steps、策略从不主动 terminate。全零地板上
# 四臂无法区分,E5 问不出结论。
#
# 本脚本把测量点从"整条任务成功"下移到**决策步的动作**:给定专家前缀
# (teacher forcing)与同一当前屏,只改喂进去的历史子集,看冻结 executor
# 输出的动作是否命中专家动作。这是**配对**设计(同状态同前缀,只换记忆),
# 统计上比闭环干净得多,且不受地板效应影响。
#
# **口径声明(重要)**:step-level 正确率在本路线里**只能作诊断**,
# 路线文档 §9 明令禁止把它用作训练奖励或 selector 的监督信号 ——
# 本脚本只输出诊断表,不产出任何可用于训练的标签。
# 闭环 E5 的正式判定留给 Phase 2 的 policy_mem_sft(--adapter)。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def mcnemar_p(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def boot_ci(diffs: list[int], reps: int = 1999, seed: int = 12345):
    r = random.Random(seed)
    n = len(diffs)
    if not n:
        return 0.0, 0.0
    means = sorted(sum(r.choice(diffs) for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train",
                   choices=["train", "syn_iid", "syn_ood"])
    p.add_argument("--n-tasks", type=int, default=200)
    p.add_argument("--task-seed", type=int, default=101)
    p.add_argument("--arms", nargs="+",
                   default=["oracle", "recent2", "random2", "none"])
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--adapter", type=Path, default=None,
                   help="给了就评 policy_mem_sft;不给 = 冻结基线")
    p.add_argument("--shot-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=25.0)
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    args = p.parse_args()

    import sys
    import torch

    from causalcache.osworld_gui_owl import _TOOL_SPEC, GUIOwlOSWorldRuntime
    from causalcache_agentic import tasks as tasks_mod
    from causalcache_agentic.env import GUIEnv
    from causalcache_agentic.policy_io import (
        HistoryFrameBank,
        PolicyInputBuilder,
        canonical_action,
    )
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rl_oracle_enumerate import action_correct, parse_tool_call
    for cand in (Path(__file__).resolve().parents[3] / "code" / "scripts",
                 Path("/data/osworld/CausalCache/code/scripts")):
        if cand.exists():
            sys.path.insert(0, str(cand))
            break

    ds = tasks_mod.make_dataset(args.split, args.n_tasks, args.task_seed)
    ds = [t for i, t in enumerate(ds) if i % args.shard_count == args.shard_index]
    print(json.dumps({"tasks": len(ds), "split": args.split}), flush=True)

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir, expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device, effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=128)
    model, device, proc = runtime.model, runtime.device, runtime.processor
    if args.adapter is not None:
        from train_success_sft_lora import inject_lora, load_lora_state_dict
        b = torch.load(args.adapter, map_location="cpu", weights_only=False)
        wrapped = inject_lora(
            model, rank=int(b["rank"]), alpha=int(b["alpha"]),
            target_modules=tuple(str(b.get(
                "target_modules", "q_proj,k_proj,v_proj,o_proj")).split(",")),
            torch=torch,
            last_layer_count=int(b["last_layers"]) if b.get("last_layers") else None)
        load_lora_state_dict(wrapped, b["state"])
        print(json.dumps({"adapter": str(args.adapter), "epoch": b.get("epoch"),
                          "modules": len(wrapped)}), flush=True)

    # note (luojiaxuan): builder 强制 |S| == budget(Phase 3 的 selector 恒选满
    # B 张,这个约束是对的)。但 none 臂是 B=0 的对照,需要一个 budget=0 的
    # builder —— 官方协议本来就支持 kept=0 的单轮退化形态。
    builders = {b: PolicyInputBuilder(budget=b) for b in (0, args.budget)}
    env = GUIEnv()
    args.shot_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    def gen(msgs) -> dict | None:
        enc = proc.apply_chat_template(
            msgs, tools=[_TOOL_SPEC], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(device)
        pt = int(enc["input_ids"].shape[1])
        with torch.inference_mode():
            gt = runtime.generation_tokens
            o = model.generate(**enc, do_sample=False, max_new_tokens=128,
                               eos_token_id=gt.tool_call_close_token_id,
                               pad_token_id=gt.pad_token_id,
                               suppress_tokens=list(gt.standard_eos_token_ids),
                               num_beams=1, num_return_sequences=1)
        return parse_tool_call(proc.batch_decode(
            o[:, pt:], skip_special_tokens=False)[0])

    done = set()
    if args.output.exists():
        for l in args.output.open(encoding="utf-8"):
            l = l.strip()
            if l:
                done.add(json.loads(l)["task_id"])
    n = 0
    with args.output.open("a", encoding="utf-8") as sink:
        for task in ds:
            if task.task_id in done:
                continue
            mp = task.memory_probe or {}
            dec = mp.get("decision_step")
            if dec is None:
                continue
            dec = int(dec)
            try:
                # 专家前缀重放到决策步,逐步落盘截图(与 rollout 同一渲染路径)
                st = env.reset(task)
                bank = HistoryFrameBank()
                actions: list[dict] = []
                for k in range(dec + 1):
                    path = str(args.shot_dir / f"{task.task_id}_s{k:02d}.png")
                    env.render(path)
                    bank.add(k, path)
                    if k < dec:
                        a = dict(task.expert_actions[k])
                        actions.append(a)
                        st, _, _ = env.step(a)
                gold = dict(task.expert_actions[dec])
                cands = bank.candidates(dec)
                if len(cands) < args.budget:
                    continue
                recent = tuple(cands[-args.budget:])
                req = [i for i in (mp.get("required_steps") or []) if i in cands]
                oracle = tuple(sorted(set(
                    (req[-args.budget:] if len(req) >= args.budget
                     else list(req) + [c for c in reversed(recent)
                                       if c not in req])[: args.budget])))
                rnd = tuple(sorted(rng.sample(cands, args.budget)))
                subsets = {"oracle": oracle, "recent2": recent, "random2": rnd,
                           "none": ()}
                row: dict[str, Any] = {
                    "task_id": task.task_id, "template_id": task.template_id,
                    "family": task.family, "regime": task.regime,
                    "decision_step": dec, "candidates": list(cands),
                    "gold": gold, "subsets": {k: list(v) for k, v in subsets.items()},
                }
                for arm in args.arms:
                    sub = subsets[arm]
                    msgs = builders[len(sub)].build(
                        task.instruction, actions, sub, bank, dec)
                    pred = gen(msgs)
                    row[f"{arm}_pred"] = pred
                    row[f"{arm}_correct"] = int(action_correct(
                        canonical_action(pred) if pred else pred, gold,
                        tolerance=args.tolerance))
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                n += 1
                if n % 20 == 0:
                    print(json.dumps({"done": n}), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"task_error": f"{type(exc).__name__}: {exc}"[:200],
                                  "task_id": task.task_id}), flush=True)
                continue
    print(json.dumps({"evaluated": n, "finished": True}), flush=True)


if __name__ == "__main__":
    main()
