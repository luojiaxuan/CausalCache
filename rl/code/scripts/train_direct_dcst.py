#!/usr/bin/env python3
"""Direct DCST 训练器(设计冻结版,rl/docs/selector_direct_dcst_frozen_20260812.md)。

# note (luojiaxuan): 关键纪律:
#   * batch 单位 = 整个 state(全部已标注子集一起进损失,先态内平均);
#   * 损失 = L_base(q 头)+ L_cond(条件 rescue/harm)+ L_deploy
#     (结构化 SmoothMax margin,温度退火),α=β=1;
#   * 模型选择 = 内层 (selector−recent) paired diff(τ 在内层网格校准),
#     禁 pair AUC/BCE;早停 + 最优检查点落盘(含 τ);
#   * 文本上下文用 GUI-Owl 冻结词嵌入(safetensors 只捞 embed_tokens,
#     不加载策略模型本体);
#   * candidate dropout 增广(非 recent 帧 p=0.15;含被删帧的子集同弃);
#     候选置换增广省略 —— 本实现无候选序位置编码,结构上已置换等变。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def smoothmax(v, t: float):
    import torch
    return t * torch.logsumexp(v / t, dim=0)


def load_states(labels: Path, token_dir: Path):
    table = {}
    for line in labels.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        subs = {tuple(sorted(a["s"])): bool(a["c"]) for a in d.get("all", [])
                if len(a["s"]) == 2}
        vals = list(subs.values())
        if not (vals and any(vals) and not all(vals)):
            continue
        if not (token_dir / f"{d['dp_id']}.pt").exists():
            continue
        table[d["dp_id"]] = {"subs": subs, "b": bool(d["recent_correct"])}
    return table


def frozen_embed(model_dir: Path, device):
    """只捞 embed_tokens.weight,不加载策略模型。"""
    import torch
    from safetensors import safe_open
    idx_file = model_dir / "model.safetensors.index.json"
    if idx_file.exists():
        wm = json.loads(idx_file.read_text())["weight_map"]
        name = next(k for k in wm if k.endswith("embed_tokens.weight"))
        shard = model_dir / wm[name]
    else:
        shard = model_dir / "model.safetensors"
        with safe_open(str(shard), framework="pt") as f:
            name = next(k for k in f.keys() if k.endswith("embed_tokens.weight"))
    with safe_open(str(shard), framework="pt") as f:
        w = f.get_tensor(name)
    return w.to(device=device, dtype=torch.float32).requires_grad_(False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--token-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--torch-seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--warmup-epochs", type=int, default=2,
                   help="先只训概率头(L_base+L_cond),再加 L_deploy")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--cand-dropout", type=float, default=0.15)
    p.add_argument("--inner-frac", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--limit-states", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260809)
    args = p.parse_args()

    import sys
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from causalcache_rl.direct_dcst import DirectDCST
    from causalcache.agentnet_desktop_official import official_step_forms

    table = load_states(args.labels, args.token_dir)
    ids = sorted(table)
    random.Random(args.seed).shuffle(ids)
    nh = int(len(ids) * 0.2)
    lo, hi = args.fold * nh, (args.fold + 1) * nh
    hold = set(ids[lo:hi])
    train_all = [i for i in ids if i not in hold]
    k_inner = int(len(train_all) * (1 - args.inner_frac))
    train_ids, inner_ids = train_all[:k_inner], train_all[k_inner:]
    if args.limit_states:
        train_ids = train_ids[: args.limit_states]
        inner_ids = inner_ids[: max(8, args.limit_states // 4)]
    print(json.dumps({"winnable": len(ids), "train": len(train_ids),
                      "inner": len(inner_ids), "fold": args.fold}), flush=True)

    dev = torch.device(args.device)
    emb = frozen_embed(args.model_dir, dev)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.model_dir),
                                        local_files_only=True)

    meta: dict[str, dict] = {}
    for line in args.manifest.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d["dp_id"] not in table:
            continue
        s = int(d["step"])
        screen = tuple(int(x) for x in d["screen_size"])
        try:
            steps = [official_step_forms(h, screen_size=screen)
                     for h in d["history"]]
        except (ValueError, KeyError):
            continue
        cands = [j for j in range(1, s - 1) if steps[j].full_response]
        if len(cands) < 2:
            continue
        ids_of = {j: k for k, j in enumerate(cands)}
        instr = tok(d["instruction"][:2000], add_special_tokens=False,
                    return_tensors="pt")["input_ids"][0][:256]
        seg_ids = [instr]
        for t in range(1, s):
            line_t = f"Step{t}: {steps[t - 1].action_line}"
            seg_ids.append(tok(line_t, add_special_tokens=False,
                               return_tensors="pt")["input_ids"][0][:64])
        act_ids = {j: tok(steps[j - 1].action_line[:200],
                          add_special_tokens=False,
                          return_tensors="pt")["input_ids"][0][:64]
                   for j in cands}
        meta[d["dp_id"]] = {"cands": cands, "ids_of": ids_of, "step": s,
                            "segs": seg_ids, "act": act_ids}
    usable = [i for i in ids if i in meta]
    print(json.dumps({"with_meta": len(usable)}), flush=True)

    torch.manual_seed(args.torch_seed)
    model = DirectDCST().to(dev)
    n_par = sum(p_.numel() for p_ in model.parameters())
    print(json.dumps({"params": n_par}), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    opt.zero_grad(set_to_none=True)

    def load_state(dp: str, drop: float = 0.0, rng: random.Random | None = None):
        m = meta[dp]
        e = torch.load(args.token_dir / f"{dp}.pt", map_location="cpu",
                       weights_only=False)
        cands = e["cands"]
        assert cands == m["cands"], f"候选池错位 {dp}"
        keep = list(range(len(cands)))
        if drop and rng and len(cands) > 4:
            rec = {len(cands) - 2, len(cands) - 1}
            keep = [k for k in keep
                    if k in rec or rng.random() > drop]
        toks = [e["toks"][k].to(dev).float() for k in keep]
        cur = e["ctx"].to(dev).float()
        segs = [emb[s.to(dev)] for s in m["segs"]]
        ages = [m["step"] - 1 - cands[k] for k in keep]
        acts = [emb[m["act"][cands[k]].to(dev)] for k in keep]
        remap = {k: i for i, k in enumerate(keep)}
        recent_local = [remap[len(cands) - 2], remap[len(cands) - 1]]
        subs = []
        for s_, y in table[dp]["subs"].items():
            loc = tuple(sorted(remap[m["ids_of"][j]] for j in s_
                               if m["ids_of"][j] in remap))
            if len(loc) == len(s_):
                subs.append((loc, y))
        return toks, cur, segs, ages, acts, recent_local, subs, table[dp]["b"]

    def state_losses(dp, ep, rng):
        toks, cur, segs, ages, acts, rec, subs, b = load_state(
            dp, args.cand_dropout, rng)
        if len(subs) < 2 or not any(y for _, y in subs) \
                or all(y for _, y in subs):
            raise ValueError("degenerate after dropout")
        st = model.encode_state(toks, cur, segs, ages, acts, rec)
        q_logit = model.recent_failure(st)
        sub_list = [s_ for s_, _ in subs]
        y = torch.tensor([float(v) for _, v in subs], device=dev)
        r_log, h_log = model.score_subsets(st, sub_list)
        l_base = torch.nn.functional.binary_cross_entropy_with_logits(
            q_logit, torch.tensor(1.0 - b, device=dev))
        if not b:
            l_cond = torch.nn.functional.binary_cross_entropy_with_logits(
                r_log, y)
        else:
            l_cond = torch.nn.functional.binary_cross_entropy_with_logits(
                h_log, 1.0 - y)
        q = torch.sigmoid(q_logit)
        g = q * torch.sigmoid(r_log) - (1 - q) * torch.sigmoid(h_log)
        rset = frozenset(rec)
        keepmask = torch.tensor([frozenset(s_) != rset for s_ in sub_list],
                                device=dev)
        t_sm = max(0.15, 1.0 - 0.85 * ep / max(args.epochs - 1, 1))
        pos = g[(y > 0.5) & keepmask]
        neg = g[(y < 0.5) & keepmask]
        m_ = args.margin
        if not b:
            gp = smoothmax(pos, t_sm) if pos.numel() else g.new_tensor(-1.0)
            gn = smoothmax(neg, t_sm) if neg.numel() else g.new_tensor(-1.0)
            l_dep = (torch.nn.functional.softplus(m_ - gp)
                     + torch.nn.functional.softplus(m_ + gn - gp))
        else:
            gn = smoothmax(neg, t_sm) if neg.numel() else g.new_tensor(-1.0)
            l_dep = torch.nn.functional.softplus(m_ + gn)
        beta = 0.0 if ep < args.warmup_epochs else args.beta
        return l_base + args.alpha * l_cond + beta * l_dep

    @torch.no_grad()
    def inner_eval():
        model.eval()
        rows = []
        for dp in inner_ids:
            if dp not in meta:
                continue
            try:
                toks, cur, segs, ages, acts, rec, subs, b = load_state(dp)
            except (ValueError, KeyError, OSError, AssertionError):
                continue
            st = model.encode_state(toks, cur, segs, ages, acts, rec)
            q = torch.sigmoid(model.recent_failure(st))
            rset = frozenset(rec)
            cand = [(s_, y_) for s_, y_ in subs if frozenset(s_) != rset]
            if not cand:
                continue
            r_log, h_log = model.score_subsets(st, [s_ for s_, _ in cand])
            g = (q * torch.sigmoid(r_log)
                 - (1 - q) * torch.sigmoid(h_log)).tolist()
            k = max(range(len(g)), key=lambda i: g[i])
            rows.append({"b": int(b), "gmax": g[k], "y": int(cand[k][1])})
        model.train()
        best = None
        for tau in [i / 50 for i in range(26)]:
            acc_s = sum(r["y"] if r["gmax"] > tau else r["b"] for r in rows)
            acc_r = sum(r["b"] for r in rows)
            w = sum(1 for r in rows if r["gmax"] > tau and r["y"] and not r["b"])
            l_ = sum(1 for r in rows if r["gmax"] > tau and not r["y"] and r["b"])
            d = (acc_s - acc_r) / max(len(rows), 1)
            if best is None or (d, w - l_) > (best[0], best[1]):
                best = (d, w - l_, tau, w, l_)
        return {"inner_diff_pp": round(100 * best[0], 2), "W": best[3],
                "L": best[4], "tau": best[2], "n": len(rows)}

    args.output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed + 13 + args.torch_seed)
    best_diff, best_ep, since = None, -1, 0
    seen = 0
    skipped: dict[str, int] = {}
    for ep in range(args.epochs):
        order = list(train_ids)
        random.Random(args.seed + ep).shuffle(order)
        run_loss = 0.0
        for dp in order:
            if dp not in meta:
                continue
            try:
                loss = state_losses(dp, ep, rng)
                (loss / args.grad_accum).backward()
                seen += 1
                run_loss += float(loss.detach())
                if seen % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            except (ValueError, KeyError, OSError, RuntimeError,
                    AssertionError) as exc:
                k = type(exc).__name__ if not str(exc) else str(exc)[:40]
                skipped[k] = skipped.get(k, 0) + 1
                opt.zero_grad(set_to_none=True)
                continue
        ev = inner_eval()
        print(json.dumps({"epoch": ep, "mean_loss": round(
            run_loss / max(len(order), 1), 4), **ev,
            "skipped": skipped}, ensure_ascii=False), flush=True)
        key = (ev["inner_diff_pp"], ev["W"] - ev["L"])
        if best_diff is None or key > best_diff:
            best_diff, best_ep, since = key, ep, 0
            torch.save({"state": model.state_dict(), "tau": ev["tau"],
                        "epoch": ep, "fold": args.fold,
                        "torch_seed": args.torch_seed,
                        "inner": ev}, args.output_root / "best.pt")
        else:
            since += 1
            if since >= args.patience:
                print(json.dumps({"early_stop": ep}), flush=True)
                break
    print(json.dumps({"final": True, "best_epoch": best_ep,
                      "best_inner": best_diff[0] if best_diff else None,
                      "skipped": skipped}, ensure_ascii=False))
    n_skip = sum(skipped.values())
    if n_skip > 0.15 * max(seen + n_skip, 1):
        raise SystemExit(f"FAILED: 跳过率 {n_skip}/{seen + n_skip} 超过 15%")


if __name__ == "__main__":
    main()
