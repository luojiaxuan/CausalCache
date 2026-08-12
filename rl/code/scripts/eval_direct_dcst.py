#!/usr/bin/env python3
"""Direct DCST 外层折部署评测:全 C(n,2) 打分 → τ 门 → 查表(缺口落盘补测)。

# note (luojiaxuan): 与全部历史评测同一把尺(同总体/划分/查表/配对统计)。
# 模型可以给未标注子集打分 —— argmax 在全组合空间;选中对不在表内则落盘
# missing_pairs 交冻结策略补测(rl_score_chosen_subsets),n_unmeasured=0
# 才算完整口径。τ 来自训练时内层校准,外层不动。
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--token-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--domain-json", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--draft-files", nargs="*", type=Path, default=[])
    p.add_argument("--limit-states", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260809)
    args = p.parse_args()

    import sys
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from causalcache_rl.direct_dcst import DirectDCST
    from causalcache.agentnet_desktop_official import official_step_forms
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_direct_dcst import frozen_embed, load_states

    table = load_states(args.labels, args.token_dir)
    ids = sorted(table)
    random.Random(args.seed).shuffle(ids)
    nh = int(len(ids) * 0.2)
    hold = ids[args.fold * nh:(args.fold + 1) * nh]

    dom = json.loads(args.domain_json.read_text())
    dev = torch.device(args.device)
    emb = frozen_embed(args.model_dir, dev)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.model_dir),
                                        local_files_only=True)
    bundle = torch.load(args.checkpoint, map_location="cpu",
                        weights_only=False)
    cfg = bundle.get("cfg", {})
    model = DirectDCST(d=cfg.get("dim", 512),
                       k_spatial=cfg.get("k_spatial", 96),
                       k_global=cfg.get("k_global", 32),
                       dropout=cfg.get("dropout", 0.1),
                       draft_dim=cfg.get("draft_dim", 0)).to(dev)
    model.load_state_dict(bundle["state"])
    model.eval()
    tau = float(bundle["tau"])
    print(json.dumps({"holdout": len(hold), "fold": args.fold, "tau": tau,
                      "ckpt_epoch": bundle.get("epoch"),
                      "inner": bundle.get("inner")}), flush=True)

    draft_feats: dict[str, list[float]] = {}
    if bundle.get("cfg", {}).get("draft_dim"):
        from rl_e0_draft_gate_probe import features as draft_features
        for f in args.draft_files:
            for line in f.open(encoding="utf-8"):
                line = line.strip()
                if line:
                    r_ = json.loads(line)
                    draft_feats[r_["dp_id"]] = draft_features(r_)
        print(json.dumps({"draft_feats": len(draft_feats)}), flush=True)

    t2d = {}
    meta = {}
    for line in args.manifest.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        t2d[d["dp_id"]] = dom.get(d.get("task_id"), "?")
        if d["dp_id"] in set(hold):
            meta[d["dp_id"]] = d

    done = set()
    if args.output.exists():
        for l in args.output.open(encoding="utf-8"):
            l = l.strip()
            if l:
                done.add(json.loads(l)["dp"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    skipped: dict[str, int] = {}
    with args.output.open("a", encoding="utf-8") as sink, torch.no_grad():
        for dp in hold:
            if args.limit_states and n >= args.limit_states:
                break
            rec = meta.get(dp)
            if rec is None or dp in done:
                continue
            try:
                s = int(rec["step"])
                screen = tuple(int(x) for x in rec["screen_size"])
                steps = [official_step_forms(h, screen_size=screen)
                         for h in rec["history"]]
                cands = [j for j in range(1, s - 1) if steps[j].full_response]
                e = torch.load(args.token_dir / f"{dp}.pt",
                               map_location="cpu", weights_only=False)
                if e["cands"] != cands:
                    raise ValueError("candidate mismatch")
                toks = [t_.to(dev).float() for t_ in e["toks"]]
                cur = e["ctx"].to(dev).float()
                instr = tok(rec["instruction"][:2000], add_special_tokens=False,
                            return_tensors="pt")["input_ids"][0][:256]
                segs = [emb[instr.to(dev)]]
                for t_ in range(1, s):
                    li = tok(f"Step{t_}: {steps[t_ - 1].action_line}",
                             add_special_tokens=False,
                             return_tensors="pt")["input_ids"][0][:64]
                    segs.append(emb[li.to(dev)])
                acts = [emb[tok(steps[j - 1].action_line[:200],
                                add_special_tokens=False,
                                return_tensors="pt")["input_ids"][0][:64]
                            .to(dev)] for j in cands]
                ages = [s - 1 - j for j in cands]
                nloc = len(cands)
                recent = [nloc - 2, nloc - 1]
                st = model.encode_state(toks, cur, segs, ages, acts, recent)
                dfeat = (torch.tensor(draft_feats[dp], device=dev)
                         if dp in draft_feats else None)
                q = float(torch.sigmoid(model.recent_failure(st, dfeat)))
                all_subs = [t_ for t_ in itertools.combinations(range(nloc), 2)
                            if set(t_) != set(recent)]
                gains = []
                for k0 in range(0, len(all_subs), 128):
                    chunk = all_subs[k0:k0 + 128]
                    r_log, h_log = model.score_subsets(st, chunk)
                    g = (q * torch.sigmoid(r_log)
                         - (1 - q) * torch.sigmoid(h_log))
                    gains += [float(x) for x in g]
                kbest = max(range(len(gains)), key=lambda i: gains[i])
                move = gains[kbest] > tau
                loc = all_subs[kbest] if move else tuple(recent)
                chosen = tuple(sorted(cands[i] for i in loc))
                rpair = tuple(sorted(cands[i] for i in recent))
                subs = table[dp]["subs"]
                sink.write(json.dumps({
                    "dp": dp, "domain": t2d.get(dp, "?"),
                    "q_fail": round(q, 4), "gmax": round(gains[kbest], 4),
                    "chosen": list(chosen), "moved": int(move),
                    "c_sel": (int(subs[chosen]) if chosen in subs else None),
                    "c_rec": int(subs[rpair]),
                    "fallback": 0}, ensure_ascii=False) + "\n")
                sink.flush()
                n += 1
                if n % 25 == 0:
                    print(json.dumps({"done": n}), flush=True)
            except (ValueError, KeyError, OSError, RuntimeError,
                    AssertionError) as exc:
                k = type(exc).__name__ if not str(exc) else str(exc)[:40]
                skipped[k] = skipped.get(k, 0) + 1
                continue
    print(json.dumps({"evaluated": n, "skipped": skipped, "finished": True},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
