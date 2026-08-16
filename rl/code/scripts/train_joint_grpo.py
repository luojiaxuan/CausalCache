#!/usr/bin/env python3
"""Phase 4A:joint GRPO —— selector 与 executor adapter 同时被任务奖励塑造。

# note (luojiaxuan): 路线见 agentic_memory_rl_roadmap_20260813.md §5;
# 做这一步的直接依据是 Phase 3 的诊断(审计 §4.7):闭环瓶颈已不在 selector
# (它追平了两种独立构造的 oracle),而在 executor —— 需老帧任务里 28.4% 的
# rollout **证据从未上屏**(探索失败,记忆天花板=0),证据上屏的部分也只有
# 20.6% 走完全程(而 teacher-forced 给对帧时是 98.6%)。
#
# **本阶段的设计约束(逐条对应路线 §5)**:
#   * 初始化 = policy_mem_sft(executor)+ Phase 3 最佳 selector,
#     **不从两个随机组件同时学**;
#   * 4A 只解冻 selector 全部参数 + executor 的**后 N 层 LoRA**;
#     视觉编码器与多数 backbone 保持冻结(历史帧特征缓存因此仍然有效);
#   * 一步的联合动作是 (S_t, a_t),log π = log π_sel + log π_pol,
#     **共用同一个 trajectory-level group advantage**;
#   * 奖励只有终局 0/1;
#   * 稳定手段:两组独立学习率、对 policy_mem_sft 的可选 KL、
#     交替更新(--alternate)、grad-norm/KL/entropy 分开监控。
#
# **命门:logprob 一致性自检**。selector 侧那条检查(max|Δ|=2.6e-07)救过命;
# executor 侧的分布必须与采样时**逐位同构**(同温度、同 suppress 屏蔽),
# 否则 ratio 全错而日志看上去一切正常。启动时强制自检,超阈直接退出。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def load_groups(paths: list[Path], min_group: int = 2) -> list[dict]:
    """按 (task_id, group_id) 聚合 rollout。"""
    buckets: dict[tuple[str, int], list[dict]] = {}
    for p in paths:
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # note (luojiaxuan): group_id 在 rollout schema 里是**字符串**
            # (形如 "<template>::<regime>::<seed>"),不要强转 int —— 第一次
            # 就是这么崩的。原样当 key 用即可。
            buckets.setdefault((d["task_id"], str(d.get("group_id", ""))),
                               []).append(d)
    out = []
    for (tid, gid), rolls in sorted(buckets.items()):
        if len(rolls) < min_group:
            continue
        out.append({"task_id": tid, "group_id": gid, "rollouts": rolls})
    return out


def advantages(rewards: list[float], eps: float = 1e-6):
    m = sum(rewards) / len(rewards)
    var = sum((r - m) ** 2 for r in rewards) / len(rewards)
    sd = math.sqrt(var)
    if sd < 1e-9:
        return None
    return [(r - m) / (sd + eps) for r in rewards]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", nargs="+", type=Path, required=True)
    p.add_argument("--selector-ckpt", type=Path, required=True)
    p.add_argument("--executor-adapter", type=Path, required=True,
                   help="policy_mem_sft 的 LoRA(Phase 2 产物),作为起点与 π_ref")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--out-selector", type=Path, required=True)
    p.add_argument("--out-adapter", type=Path, required=True)
    p.add_argument("--exec-last-layers", type=int, default=8,
                   help="4A:executor 只解冻后 N 层的 LoRA(0=全层,谨慎)")
    p.add_argument("--lr-selector", type=float, default=2e-4)
    p.add_argument("--lr-executor", type=float, default=2e-5)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--kl-coef", type=float, default=0.0,
                   help=">0 时对 policy_mem_sft 做 KL 正则(需额外一次前向)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--alternate", type=int, default=0,
                   help=">0 时每 N 个优化步只更新一侧(交替更新)")
    p.add_argument("--executor-temperature", type=float, default=1.0)
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--limit-groups", type=int, default=0)
    p.add_argument("--old-cache", type=Path, default=None,
                   help="log π_old 预扫缓存(默认落在 out-selector 同目录)")
    p.add_argument("--train-dropout", action="store_true",
                   help="前向开 dropout(与 Phase 3 同名开关);默认关,理由见代码")
    p.add_argument("--consistency-checks", type=int, default=24)
    p.add_argument("--consistency-tol", type=float, default=0.02,
                   help="**per-token** 结构性阈值:1e-3 量级=bf16 噪声,"
                        "1e0 量级=结构错配,取 0.02 在两者之间")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260816)
    args = p.parse_args()

    import sys
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    for cand in (Path(__file__).resolve().parents[1],
                 Path("/data/agentic/code")):
        if (cand / "causalcache_agentic").exists():
            sys.path.insert(0, str(cand))
            break
    for cand in (Path(__file__).resolve().parents[3] / "code" / "scripts",
                 Path("/data/osworld/CausalCache/code/scripts")):
        if cand.exists():
            sys.path.insert(0, str(cand))
            break

    from causalcache.osworld_gui_owl import _TOOL_SPEC, GUIOwlOSWorldRuntime
    from causalcache_agentic.policy_io import HistoryFrameBank, PolicyInputBuilder
    from causalcache_agentic.selector_model import (
        SelectorStateBuilder,
        SubsetSelectorPolicy,
    )
    from train_success_sft_lora import inject_lora, load_lora_state_dict, lora_state_dict

    groups = load_groups(list(args.rollouts))
    if args.limit_groups:
        groups = groups[: args.limit_groups]
    n_eff = sum(1 for g in groups
                if advantages([float(bool(r.get("success")))
                               for r in g["rollouts"]]) is not None)
    print(json.dumps({"groups": len(groups), "effective": n_eff,
                      "frac_effective": round(n_eff / max(len(groups), 1), 4)},
                     ensure_ascii=False), flush=True)

    dev = torch.device(args.device)
    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir, expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device, effective_visual_tokens_per_image=2560,
        max_new_tokens=128)
    model, proc = runtime.model, runtime.processor
    gt = runtime.generation_tokens
    bundle = torch.load(args.executor_adapter, map_location="cpu",
                        weights_only=False)
    wrapped_all = inject_lora(
        model, rank=int(bundle["rank"]), alpha=int(bundle["alpha"]),
        target_modules=tuple(str(bundle.get(
            "target_modules", "q_proj,k_proj,v_proj,o_proj")).split(",")),
        torch=torch,
        last_layer_count=(int(bundle["last_layers"])
                          if bundle.get("last_layers") else None))
    load_lora_state_dict(wrapped_all, bundle["state"])
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # 4A:只解冻后 N 层的 LoRA;其余(含视觉塔)保持冻结
    import re as _re
    layer_re = _re.compile(r"\.layers\.(\d+)\.")
    layers = sorted({int(m.group(1)) for name in wrapped_all
                     if (m := layer_re.search(f".{name}."))})
    keep = set(layers[-args.exec_last_layers:]) if args.exec_last_layers else set(layers)
    exec_params = []
    for name, w in wrapped_all.items():
        m = layer_re.search(f".{name}.")
        train_it = (m is not None and int(m.group(1)) in keep)
        for q in (w.lora_a, w.lora_b):
            q.requires_grad_(bool(train_it))
            if train_it:
                exec_params.append(q)
    print(json.dumps({"lora_modules": len(wrapped_all),
                      "trainable_exec_params": sum(q.numel() for q in exec_params),
                      "unfrozen_layers": sorted(keep)}), flush=True)

    selector = SubsetSelectorPolicy.load(str(args.selector_ckpt), map_location=dev)
    # note (luojiaxuan): 与 Phase 3 训练器同口径 —— **默认 eval**。开着 dropout
    # 会让 ratio 在任何更新之前就 ≠1(纯噪声),clip 大量触发、更新方向被污染。
    # 我第一版写的 .train() 正是踩了这个已经记录在案的坑。
    selector.to(dev)
    selector.train(bool(args.train_dropout))
    sel_params = [q for q in selector.parameters() if q.requires_grad]
    # note (luojiaxuan): state 口径是 log π 的定义域,**必须**与采样端逐字同路。
    # 所以不在这里自己拼 SelectorStateBuilder(第一版就是这么写的,签名都对不上),
    # 而是直接复用 rollout_selector 的 resolve_state_factory —— 它按 kwargs
    # 超集过滤,并把 origin 落盘成 selector.state_builder 供两侧对账。
    import rollout_selector as rs
    from rollout_selector import resolve_feature_provider, resolve_state_factory

    # note (luojiaxuan): **从采样端的 argparse 取默认值**再覆盖我关心的几项,
    # 不要手拼 Namespace —— 手拼那版缺 feature_dim/feature_tokens/dummy_seed…
    # 一个个补是没有尽头的,而且采样端加字段时这里会再次悄悄失配。
    fake_args = rs.parse_args([
        "--tasks-split", "train", "--out", "/dev/null",
        "--model-dir", str(args.model_dir),
        "--snapshot-manifest", str(args.snapshot_manifest),
        "--budget", str(args.budget), "--device", args.device,
        "--feature-source", "auto", "--selector-arm", "learned",
        "--selector-ckpt", str(args.selector_ckpt),
    ])
    rec0 = groups[0]["rollouts"][0]
    sel_meta = dict(rec0.get("selector") or {})
    if sel_meta.get("kind") != "learned":
        raise SystemExit(
            f"FAILED: 这批 rollout 的 selector 臂是 {sel_meta.get('kind')!r},"
            "joint 训练要求 learned 臂(基线臂没有可训练的 log π_sel)")
    feature_provider = resolve_feature_provider(selector, fake_args)
    state_factory = resolve_state_factory(fake_args, selector, sel_meta,
                                          feature_provider)
    want = str(sel_meta.get("state_builder", "")) or None
    if want and want != state_factory.origin:
        raise SystemExit(
            f"FAILED: state 构造路径与采样端不一致(采样 {want!r} vs 训练 "
            f"{state_factory.origin!r})—— log π 定义域不同,ratio 无意义")
    print(json.dumps({"state_builder": state_factory.origin,
                      "feature_provider": feature_provider is not None}),
          flush=True)

    opt = torch.optim.AdamW(
        [{"params": sel_params, "lr": args.lr_selector},
         {"params": exec_params, "lr": args.lr_executor}], weight_decay=0.0)
    opt.zero_grad(set_to_none=True)
    builder = PolicyInputBuilder(budget=args.budget)
    suppress = list(gt.standard_eos_token_ids)

    def exec_logprob(step: dict, rec: dict, task_instr: str) -> "torch.Tensor":
        """teacher-forced 重算 log π_pol(与采样时同一套 processed 分布)。"""
        toks = step.get("action_tokens")
        if not toks:
            return None
        bank = HistoryFrameBank()
        for s in rec["steps"]:
            if int(s["step"]) <= int(step["step"]):
                bank.add(int(s["step"]), s["screenshot"])
        hist = [dict(s["action"]) for s in rec["steps"]
                if int(s["step"]) < int(step["step"])]
        msgs = builder.build(task_instr, hist, step["chosen_subset"], bank,
                             int(step["step"]))
        enc = proc.apply_chat_template(
            msgs, tools=[_TOOL_SPEC], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(dev)
        tgt = torch.tensor([toks], device=dev)
        input_ids = torch.cat([enc["input_ids"], tgt], dim=1)
        kw = {k: v for k, v in enc.items() if k not in ("input_ids",
                                                        "attention_mask")}
        if "mm_token_type_ids" in kw:
            kw["mm_token_type_ids"] = torch.cat(
                [kw["mm_token_type_ids"], torch.zeros_like(tgt)], dim=1)
        # note (luojiaxuan): **必须显式 use_cache=False** —— HF 在 use_cache=True
        # 时会静默关掉梯度检查点(只发一条 warning),9.3k token 的序列于是把全部
        # 激活留在显存里:实测 61.5 GB 直接 OOM,而症状看上去只是"卡不够大"。
        out = model(input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    logits_to_keep=len(toks) + 1, **kw)
        logits = out.logits[0, :-1].float() / max(args.executor_temperature, 1e-6)
        if suppress:
            logits[:, suppress] = float("-inf")
        lp = torch.log_softmax(logits, dim=-1)
        return lp.gather(-1, tgt[0].unsqueeze(-1)).sum()

    def _okey(rec: dict, st: dict) -> str:
        """log π_old 的**全局唯一**键。

        # note (luojiaxuan): 这里踩过一个代价很大的坑 —— 最初用
        # (rollout_id, step) 做键,而 ``rollout_id`` 只是**组内序号 0..7**,
        # 不是全局唯一。384 条 rollout 的 4341 个 (id, step) 组合被压成 128 个,
        # 97% 的 old 被别的 rollout 覆盖,训练时读到的是**另一条轨迹**的 logπ,
        # ratio 于是爆到 1e+26。更阴的是它伪装成"步长过大":76% 的步近乎确定
        # (logπ≈-0.001),两条 rollout 的值都≈0,键错了 ratio 仍≈1,所以中位数
        # 停在 0.9993 看不出问题,只有在**不确定的步**上才咬人 —— 而那正是唯一
        # 携带学习信号的步。故键必须含 task_id 与 group_id。
        """
        return (f"{rec['task_id']}|{rec.get('group_id','')}|"
                f"{rec['rollout_id']}|{int(st['step'])}")

    def build_state(rec: dict, st: dict):
        """按采样端同一条路径重建 selector state(预扫与训练共用一份实现)。"""
        bank = HistoryFrameBank(feature_provider=feature_provider)
        for s in rec["steps"]:
            if int(s["step"]) <= int(st["step"]):
                bank.add(int(s["step"]), s["screenshot"])
        return state_factory(
            selector=selector, task_instruction=rec.get("instruction", ""),
            history_actions=[dict(s["action"]) for s in rec["steps"]
                             if int(s["step"]) < int(st["step"])],
            history_lines=[s["action_line"] for s in rec["steps"]
                           if int(s["step"]) < int(st["step"])],
            bank=bank, current_step=int(st["step"]),
            candidates=[int(x) for x in st["candidates"]], budget=args.budget)

    # ---- 启动预扫:在**初始权重**下重算每一步的 log π_old,并做结构性自检 ----
    # note (luojiaxuan): 这里改过一次口径,理由很实在。最初直接拿采样时记录的
    # logπ 当 old,启动自检报 max|Δ|=0.366 / mean|Δ|=0.029(整序列求和)。诊断
    # (joint_logprob_diag.py)判明是**数值噪声**:per-token mean|Δ|=0.0008,
    # 比结构性错配的量级低三个数量级,且重算两次逐位相同(确定性 0.0)。
    #
    # 但**不能就此放宽阈值**:差异不是均匀撒在所有步上,而是全部集中在 executor
    # 真正不确定的步(π≈1 时 bf16 噪声看不见;π 不确定时 logits 彼此接近,
    # log-softmax 把微小数值差放大)。实测最差一步 记录 -3.11 / 重算 -3.53,
    # Δ=-0.42 ⇒ **初始 ratio 就是 1.52,已经超出裁剪带 1.2** —— 而这些恰恰是
    # 唯一携带学习信号的步(76% 的步 π>0.9,近乎确定)。
    #
    # 故采用标准解法:old := **同一条前向路径**在初始权重下重算的值。这样初始
    # ratio 严格 =1,之后的偏移全部是真实策略变化。采样时记录的值降级为诊断量。
    # 保留的自检改成**结构性**判据(per-token,不随序列长度漂移):prompt 少一段、
    # 掩码错、错位一格会给出 ~1 nat/token,与 1e-3 的数值噪声差三个数量级。
    olds: dict[tuple[str, int], tuple[float, float]] = {}
    diffs: list[float] = []
    n_tok_tot = 0
    # note (luojiaxuan): 预扫只取决于**初始权重 + 这批数据**,与 lr/clip 等超参
    # 无关,却要占 1965 次前向(约 40 分钟)。缓存到盘上,调参重跑时直接复用;
    # 缓存键里带上两个 ckpt 路径与 rollout 文件名,换了任何一个就重算。
    cache_key = json.dumps({"sel": str(args.selector_ckpt),
                            "adp": str(args.executor_adapter),
                            "rollouts": sorted(str(x) for x in args.rollouts),
                            "budget": args.budget,
                            "temperature": args.executor_temperature},
                           sort_keys=True)
    cache_path = (args.old_cache if args.old_cache
                  else args.out_selector.parent / "old_logprob_cache.json")
    if cache_path.exists():
        blob = json.loads(cache_path.read_text())
        if blob.get("key") == cache_key:
            olds = {k: tuple(v) for k, v in blob["olds"].items()}
            print(json.dumps({"old_logprob_prescan": "从缓存加载",
                              "path": str(cache_path), "steps": len(olds)},
                             ensure_ascii=False), flush=True)
    with torch.no_grad():
        for g in ([] if olds else groups):
            if advantages([float(bool(r.get("success")))
                           for r in g["rollouts"]]) is None:
                continue                      # 无方差组不参与训练,也不必预扫
            for rec in g["rollouts"]:
                for st in rec["steps"]:
                    if st.get("policy_logprob") is None:
                        continue
                    v = exec_logprob(st, rec, rec.get("instruction", ""))
                    if v is None:
                        continue
                    sr = build_state(rec, st)
                    lp_sel = selector.logprob_of(
                        sr, [int(x) for x in st["candidates"]], args.budget,
                        tuple(int(x) for x in st["chosen_subset"]))
                    olds[_okey(rec, st)] = (float(lp_sel), float(v))
                    diffs.append(abs(float(v) - float(st["policy_logprob"])))
                    n_tok_tot += len(st["action_tokens"])
    if diffs and len(olds) != len(diffs):
        raise SystemExit(
            f"FAILED: old 缓存条目 {len(olds)} ≠ 预扫步数 {len(diffs)} —— "
            "键不唯一,会读到别的 rollout 的 logπ(这条断言就是为那个 bug 加的)")
    if not olds and not diffs:
        raise SystemExit("FAILED: rollout 里没有 policy_logprob —— "
                         "请用 --executor-sample 重新采数据")
    if diffs:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(
            {"key": cache_key,
             "olds": {k: list(v) for k, v in olds.items()}}))
    per_tok = sum(diffs) / max(n_tok_tot, 1) if diffs else 0.0
    if diffs:
        print(json.dumps({"old_logprob_prescan": {
            "steps": len(diffs), "tokens": n_tok_tot,
            "mean_abs_diff_per_step": round(sum(diffs) / len(diffs), 6),
            "mean_abs_diff_PER_TOKEN": round(per_tok, 6),
            "max_abs_diff_per_step": round(max(diffs), 6),
            "structural_tol_per_token": args.consistency_tol}},
            ensure_ascii=False), flush=True)
    if per_tok > args.consistency_tol:
        raise SystemExit(
            f"FAILED: 重算与采样的逐 token 差 {per_tok:.4f} > "
            f"{args.consistency_tol} —— 这是**结构性**错配(prompt/掩码/错位),"
            "不是 bf16 噪声,先修口径再训")

    rng = random.Random(args.seed)
    stats = {"groups_used": 0, "steps": 0, "opt_steps": 0, "skipped_groups": 0,
             "nonfinite_terms": 0, "skipped_opt_steps": 0}
    meters: dict[str, list[float]] = {}

    def meter(k: str, v: float) -> None:
        meters.setdefault(k, []).append(float(v))

    pending = 0
    for ep in range(args.epochs):
        order = list(range(len(groups)))
        rng.shuffle(order)
        for gi in order:
            g = groups[gi]
            rewards = [float(bool(r.get("success"))) for r in g["rollouts"]]
            adv = advantages(rewards)
            if adv is None:
                stats["skipped_groups"] += 1
                continue
            stats["groups_used"] += 1
            live = [(r, a) for r, a in zip(g["rollouts"], adv) if r.get("steps")]
            n_roll = len(live)
            if not n_roll:
                continue
            for rec, a_i in live:
                n_step = max(len(rec["steps"]), 1)
                for st in rec["steps"]:
                    if st.get("policy_logprob") is None:
                        continue
                    cands = [int(x) for x in st["candidates"]]
                    sr = build_state(rec, st)
                    lp_sel = selector.logprob_of(
                        sr, cands, args.budget,
                        tuple(int(x) for x in st["chosen_subset"]))
                    lp_pol = exec_logprob(st, rec, rec.get("instruction", ""))
                    if lp_pol is None:
                        continue
                    cached = olds.get(_okey(rec, st))
                    if cached is None:
                        continue
                    old = cached[0] + cached[1]
                    ratio = torch.exp(lp_sel + lp_pol - old)
                    unc = ratio * a_i
                    cl = torch.clamp(ratio, 1 - args.clip_eps,
                                     1 + args.clip_eps) * a_i
                    term = torch.min(unc, cl)
                    ent = selector.entropy(sr, cands, args.budget)
                    term = term + args.ent_coef * ent
                    # note (luojiaxuan): **最强的一条自检** —— 第一个反向之前
                    # 权重尚未动过,old 又是同一条前向路径重算的,所以此刻
                    # ratio 必须严格 =1。它不成立,就说明 old 缓存(键、state、
                    # 分布口径)有错,而这种错在损失曲线上完全看不出来。
                    r0 = float(ratio.detach())
                    if stats["steps"] == 0 and abs(r0 - 1.0) > 1e-3:
                        raise SystemExit(
                            f"FAILED: 首步 ratio={r0:.6f} ≠ 1 —— old 与重算不同源")
                    if not (math.isfinite(r0) and
                            math.isfinite(float(term.detach()))):
                        stats["nonfinite_terms"] += 1
                        continue
                    scale = 1.0 / (n_roll * n_step * max(args.grad_accum, 1))
                    (-(term) * scale).backward()
                    stats["steps"] += 1
                    meter("ratio", float(ratio.detach()))
                    # note (luojiaxuan): **两侧分开量** —— 合起来的 ratio 爆了
                    # 也说不清是 selector 的尖锐 softmax 在动,还是 executor 的
                    # 46-token 序列在累积,处方完全不同。
                    meter("ratio_sel", float(torch.exp(
                        lp_sel.detach() - cached[0])))
                    meter("ratio_pol", float(torch.exp(
                        lp_pol.detach() - cached[1])))
                    meter("clip_frac", 1.0 if abs(float(ratio.detach()) - 1) >
                          args.clip_eps else 0.0)
                    meter("entropy", float(ent.detach()))
                    meter("adv_abs", abs(a_i))
                    meter("reward", sum(rewards) / len(rewards))
            pending += 1
            if pending >= args.grad_accum:
                gn_s = torch.nn.utils.clip_grad_norm_(sel_params, 1.0)
                gn_e = torch.nn.utils.clip_grad_norm_(exec_params, 1.0)
                # note (luojiaxuan): 非有限梯度**必须丢弃整个窗口**,不能让它
                # 过 opt.step() —— clip_grad_norm_ 遇 inf/nan 时 clip 系数本身
                # 就是 0 或 nan,权重会被静默打成垃圾,而日志一切正常。
                if not (math.isfinite(float(gn_s)) and math.isfinite(float(gn_e))):
                    stats["skipped_opt_steps"] += 1
                    opt.zero_grad(set_to_none=True)
                    pending = 0
                    continue
                if args.alternate:
                    turn = (stats["opt_steps"] // args.alternate) % 2
                    for q in (sel_params if turn else exec_params):
                        q.grad = None
                opt.step()
                opt.zero_grad(set_to_none=True)
                pending = 0
                stats["opt_steps"] += 1
                meter("grad_norm_selector", float(gn_s))
                meter("grad_norm_executor", float(gn_e))
                if stats["opt_steps"] % 5 == 0:
                    print(json.dumps({"opt_step": stats["opt_steps"],
                                      **{k: round(sum(v) / len(v), 4)
                                         for k, v in meters.items() if v}},
                                     ensure_ascii=False), flush=True)

    args.out_selector.parent.mkdir(parents=True, exist_ok=True)
    selector.save(str(args.out_selector))
    torch.save({**{k: v for k, v in bundle.items() if k != "state"},
                "state": lora_state_dict(wrapped_all),
                "joint_phase": "4A",
                "exec_last_layers": args.exec_last_layers},
               args.out_adapter)
    pct = {}
    for key in ("ratio", "ratio_sel", "ratio_pol"):
        rr = sorted(meters.get(key, []))
        if not rr:
            continue
        for q in (5, 50, 95):
            pct[f"{key}_p{q}"] = round(
                rr[min(int(len(rr) * q / 100), len(rr) - 1)], 4)
        pct[f"{key}_max"] = round(rr[-1], 4)
    rr = sorted(meters.get("ratio", []))
    print(json.dumps({"final": True, **stats, **pct,
                      **{k: round(sum(v) / len(v), 4) for k, v in meters.items()
                         if v and not k.startswith("ratio")},
                      "out_selector": str(args.out_selector),
                      "out_adapter": str(args.out_adapter)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
