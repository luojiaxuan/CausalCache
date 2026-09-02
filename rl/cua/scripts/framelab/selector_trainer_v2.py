# note (luojiaxuan): selector RL trainer v2(集合能量头)。分组 = 同一
# goal(gmd5)跨 G 个 tag 的 rollout,RLOO 留一基线;决策记录自带 M 集
# 特征与位置,当前参数下按服务端完全相同的枚举序重算 slate logprob,
# 比率裁剪 + minibatch 聚合 + episode 级版本过滤,全部沿用 v1 结论
# (§4.19):版本号仅在 reload 成功后落盘。
import argparse, base64, collections, glob, itertools, json, os, time
import urllib.request

import torch

import torch.nn as nn


class Energy(nn.Module):
    """与 train_selector.Energy 同构;此处内联以免 selvenv 缺 numpy
    时被其模块级依赖拖垮。结构改动须两处同步。"""

    def __init__(self, dq, dc, h=384):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(dq, h), nn.GELU(), nn.Linear(h, h))
        self.c = nn.Sequential(nn.Linear(dc, h), nn.GELU(), nn.Linear(h, h))
        self.u = nn.Sequential(nn.Linear(2 * h + 3, h), nn.GELU(), nn.Linear(h, 1))
        self.v = nn.Sequential(nn.Linear(3 * h + 3, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, q, cands, pos, budget, subsets):
        hq = self.q(q)
        hc = self.c(cands)
        b = torch.full((1,), float(budget), device=q.device)
        es = []
        for S in subsets:
            e = q.new_zeros(())
            for i in S:
                e = e + self.u(torch.cat([hq, hc[i], pos[i], b])).squeeze()
            for i, j in itertools.combinations(S, 2):
                e = e + self.v(torch.cat([hq, hc[i], hc[j],
                                          pos[i][:1], pos[j][:1], b])).squeeze()
            es.append(e)
        return torch.stack(es)

CLIP = 0.2
ENT_COEF = 0.0
TARGET_STEPS = 8


def load_returns(path, consumed):
    rows = []
    if not os.path.exists(path):
        return {}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        rows.append(r)
    groups = collections.defaultdict(dict)   # gmd5 -> tag -> score
    for r in rows:
        groups[r["gmd5"]][r["tag"]] = float(r["score"])
    adv = {}
    for g, tags in groups.items():
        if len(tags) < 2:
            continue
        for tag, sc in tags.items():
            others = [v for t2, v in tags.items() if t2 != tag]
            key = f"{g}|{tag}"
            if key in consumed:
                continue
            adv[key] = sc - sum(others) / len(others)
    return adv


def slate_logprob(model, rec, dev):
    def _dec(x):
        if isinstance(x, str):
            return torch.frombuffer(bytearray(base64.b64decode(x)),
                                    dtype=torch.float16).to(torch.float32)
        return torch.tensor(x, dtype=torch.float32)
    q = _dec(rec["q_feat"]).to(dev)
    keep = rec["cand_set"]
    c = torch.stack([_dec(rec["feats"][str(i)]) for i in keep]).to(dev)
    p = torch.tensor([rec["pos"][str(i)] for i in keep],
                     dtype=torch.float32, device=dev)
    remap = {orig: j for j, orig in enumerate(keep)}
    subsets = list(itertools.combinations(range(len(keep)), rec.get("budget", 2)))
    es = model(q, c, p, rec.get("budget", 2), subsets)
    logits = torch.log_softmax(es, 0)
    chosen = tuple(sorted(remap[i] for i in rec["chosen"]))
    return logits[subsets.index(chosen)]


def train_round(args, device):
    consumed = set()
    if os.path.exists(args.cursor):
        consumed = set(json.load(open(args.cursor)))
    adv = load_returns(args.returns_file, consumed)
    if not adv:
        return {"sel_updates": 0, "note": "no complete groups"}

    dev = torch.device(device)
    ck = os.path.join(args.ckpt_dir, "energy_latest.pt")
    src = ck if os.path.exists(ck) else args.bundle
    b = torch.load(src, map_location=dev)
    model = Energy(b["dq"], b["dc"]).to(dev)
    model.load_state_dict(b["state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    pv_file = os.path.join(args.ckpt_dir, "selector_pv.txt")
    cur_pv = int(open(pv_file).read().strip()) if os.path.exists(pv_file) else 0

    by_ep = collections.defaultdict(list)
    for f in glob.glob(os.path.join(args.decisions_dir, "*.jsonl")):
        for line in open(f):
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            key = f"{rec.get('gmd5')}|{rec.get('tag')}"
            if key in adv and rec.get("feats"):
                by_ep[key].append(rec)

    pending, n_stale, n_mixed = [], 0, 0
    for key, recs in by_ep.items():
        vers = {r.get("pv") for r in recs}
        if vers != {cur_pv}:
            n_stale += len(recs)
            n_mixed += 1 if len(vers) > 1 else 0
            continue
        pending.extend((r, adv[key], key) for r in recs)
    import random
    random.Random(len(pending)).shuffle(pending)

    with torch.no_grad():
        drift = [abs(float(slate_logprob(model, r, dev)) - r["logp"])
                 for r, _, _ in pending[:200]]
    logp_mae = sum(drift) / len(drift) if drift else None

    MB = max(32, -(-len(pending) // TARGET_STEPS))
    losses, batch, n_clip, n_used, gn = [], [], 0, 0, None
    trained = set()

    def _step():
        nonlocal gn, batch
        if not batch:
            return
        opt.zero_grad()
        torch.stack(batch).mean().backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        batch = []

    for rec, a, key in pending:
        new_lp = slate_logprob(model, rec, dev)
        ratio = torch.exp(new_lp - torch.tensor(rec["logp"], device=dev))
        if abs(float(ratio) - 1.0) > CLIP:
            n_clip += 1
        surr = torch.min(ratio * a, ratio.clamp(1 - CLIP, 1 + CLIP) * a)
        loss = -surr
        batch.append(loss)
        losses.append(float(loss))
        n_used += 1
        trained.add(key)
        if len(batch) >= MB:
            _step()
    _step()

    torch.save({"state_dict": model.state_dict(), "dq": b["dq"], "dc": b["dc"]}, ck)
    consumed |= set(adv.keys())
    json.dump(sorted(consumed), open(args.cursor, "w"))

    reload_ok = "skipped"
    if args.service_url and n_used:
        new_pv = cur_pv + 1
        try:
            req = urllib.request.Request(
                args.service_url.rstrip("/") + "/reload",
                data=json.dumps({"path": ck, "pv": new_pv}).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=60).read()
            with open(pv_file, "w") as f:
                f.write(str(new_pv))
            reload_ok = True
        except Exception as e:  # noqa: BLE001
            reload_ok = f"FAIL:{e}"

    return {"sel_updates": n_used, "episodes_with_adv": len(adv),
            "episodes_trained": len(trained),
            "loss_mean": sum(losses) / len(losses) if losses else None,
            "clip_frac": n_clip / n_used if n_used else None,
            "logp_consistency_mae": logp_mae,
            "n_stale_dropped": n_stale, "n_mixed_version_eps": n_mixed,
            "grad_norm": float(gn) if gn is not None else None,
            "mb_size": MB, "pv": cur_pv, "reload": reload_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions-dir", required=True)
    ap.add_argument("--returns-file", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--bundle", default="/data01/jaxan/selector_feats/energy_final.pt")
    ap.add_argument("--service-url", default="http://127.0.0.1:41010")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cursor", default="")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if not args.cursor:
        args.cursor = os.path.join(args.ckpt_dir, "cursor.json")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    metrics = os.path.join(args.ckpt_dir, "metrics.jsonl")
    while True:
        try:
            m = train_round(args, args.device)
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            m = {"error": traceback.format_exc().splitlines()[-1]}
        m["t"] = time.time()
        with open(metrics, "a") as f:
            f.write(json.dumps(m) + "\n")
        print("SEL2_TRAIN", json.dumps(m), flush=True)
        if args.interval <= 0:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
