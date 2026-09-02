# note (luojiaxuan): 集合能量 selector 训练(路线 A)。
# E(S|q,B) = Σ_i u(q,i,B) + Σ_{i<j} v(q,i,j,B);q = goal⊕折叠历史文本 ⊕
# 当前帧视觉特征;候选 i = 帧视觉特征 ⊕ 来历 conclusion 文本特征 ⊕ 位置。
# 损失 = 状态内 listwise 排序(行为标签 0/1 作软目标;外审 20260901:只训
# 序不训幅值),按基数类(单/对/四元组)分别 softmax;轨迹等权。
# 切分双口径:strict = 按任务模板(目录名去尾部数字),lenient = 按轨迹。
# 评测 = 部署指标:各基数 argmax 上下文的行为正确率 vs recency/oracle/随机。
import argparse, collections, glob, itertools, json, os, random, re, sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/data01/jaxan")
from audit_analysis import parse_action, match  # noqa: E402

FEAT_DIR = "/data01/jaxan/selector_feats"
REL = "/data01/jaxan/mw/"


def load_labels():
    recs = []
    for f in sorted(glob.glob("/data01/jaxan/behav_labels_shard*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("decodes"):
                recs.append(r)
    for f in sorted(glob.glob("/data01/jaxan/behav_pilot_shard*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if not r.get("decodes"):
                continue
            ref = parse_action(r["target"])
            if ref is None:
                continue
            r2 = {k: r.get(k) for k in ("dir", "goal", "step", "target")}
            r2["slate"] = 6
            import hashlib as _h, random as _r
            _rng = _r.Random(int(_h.md5(
                f"{r['dir']}|{r['step']}".encode()).hexdigest()[:8], 16))
            r2["cand_idx"] = sorted(_rng.sample(range(0, r["step"] - 1), 6))
            r2["decodes"] = {("s" + k[1:] if k.startswith("s") else k):
                             {"t": v, "ok": match(parse_action(v), ref)}
                             for k, v in r["decodes"].items()}
            recs.append(r2)
    return recs


def template_of(d):
    return re.sub(r"\d+$", "", os.path.basename(d.rstrip("/")))


class Energy(nn.Module):
    def __init__(self, dq, dc, h=384):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(dq, h), nn.GELU(), nn.Linear(h, h))
        self.c = nn.Sequential(nn.Linear(dc, h), nn.GELU(), nn.Linear(h, h))
        self.u = nn.Sequential(nn.Linear(2 * h + 3, h), nn.GELU(), nn.Linear(h, 1))
        self.v = nn.Sequential(nn.Linear(3 * h + 3, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, q, cands, pos, budget, subsets):
        """q:[dq] cands:[n,dc] pos:[n,2] budget 标量;subsets: list of tuples."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["template", "traj"], default="template")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true",
                    help="全量数据训固定轮数,导出推理包(RL 初始化用)")
    ap.add_argument("--data-frac", type=float, default=1.0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    frames = np.load(os.path.join(FEAT_DIR, "frames.npz"))
    stext = np.load(os.path.join(FEAT_DIR, "state_text.npz"))
    recs = load_labels()
    print(f"标签状态 {len(recs)},帧特征 {len(frames.files)},文本特征 {len(stext.files)}")

    # 帧号 -> 相对路径 映射(逐轨迹)
    shots_cache = {}
    def shot_rel(d, n):
        if d not in shots_cache:
            mp = {}
            for f in glob.glob(os.path.join(d, "screenshots", "*.png")):
                m = re.search(r"-(\d+)\.png$", f)
                if m:
                    mp[int(m.group(1))] = f.replace(REL, "")
            shots_cache[d] = mp
        return shots_cache[d].get(n)

    dev = torch.device(args.device)
    # 全局居中 + 单位归一:原始特征 95% 范数是公共方向(实测跨轨迹余弦
    # 1.0000),不居中则候选间不可分。帧与文本各自估全局均值。
    fkeys = fr_files = frames.files
    fmean = np.stack([frames[f].astype(np.float32)
                      for f in fr_files[::max(1, len(fr_files) // 500)]]).mean(0)
    tkeys = stext.files
    tmean = np.stack([stext[t].astype(np.float32)
                      for t in tkeys[::max(1, len(tkeys) // 500)]]).mean(0)

    def nf(x, mean):
        v = x.astype(np.float32) - mean
        return v / (np.linalg.norm(v) + 1e-6)

    data = []
    for r in recs:
        key = f"{r['dir'].replace(REL, '')}|{r['step']}"
        qk = key + "|q"
        if qk not in stext.files:
            continue
        k = r["step"]
        cur_rel = shot_rel(r["dir"], k)
        if cur_rel is None or cur_rel not in frames.files:
            continue
        cand_files = [i + 1 for i in r.get("cand_idx", [])] or None
        if cand_files is None:
            continue
        feats, poss, keep = [], [], []
        for slot, n in enumerate(cand_files):
            fr = shot_rel(r["dir"], n)
            ck = f"{key}|c{n}"
            if fr in frames.files and ck in stext.files:
                feats.append(np.concatenate([nf(frames[fr], fmean), nf(stext[ck], tmean)]))
                poss.append([(k - n) / max(k, 1), n / max(k, 1)])
                keep.append(slot)
        if len(keep) < len(cand_files):
            continue
        q = np.concatenate([nf(stext[qk], tmean), nf(frames[cur_rel], fmean)])
        oks = {}
        for name, dv in r["decodes"].items():
            ok = dv.get("ok")
            if ok is None:
                continue
            oks[name] = ok
        data.append({"key": key, "dir": r["dir"], "task": template_of(r["dir"]),
                     "q": torch.tensor(q, dtype=torch.float32),
                     "c": torch.tensor(np.stack(feats), dtype=torch.float32),
                     "p": torch.tensor(poss, dtype=torch.float32),
                     "slate": r.get("slate", 6), "oks": oks})
    print(f"可训状态 {len(data)}")

    if args.final:
        tr, va = data, data[:200]
        traj_n = collections.Counter(d["dir"] for d in tr)
        dq = tr[0]["q"].shape[0]
        dc = tr[0]["c"].shape[1]
        model = Energy(dq, dc).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        _run_epochs(args, model, opt, tr, va, traj_n, dev)
        import numpy as _np
        torch.save({"state_dict": model.state_dict(), "dq": dq, "dc": dc},
                   "/data01/jaxan/selector_feats/energy_final.pt")
        _np.savez("/data01/jaxan/selector_feats/norm_stats.npz",
                  fmean=fmean, tmean=tmean)
        print("FINAL_BUNDLE_SAVED")
        return

    groups = collections.defaultdict(list)
    for d in data:
        groups[d["task"] if args.split == "template" else d["dir"]].append(d)
    keys = sorted(groups)
    random.Random(7).shuffle(keys)
    n_val = max(1, len(keys) // 5)
    val_keys = set(keys[:n_val])
    tr_keys = keys[n_val:]
    if args.data_frac < 1.0:
        tr_keys = tr_keys[:max(1, int(len(tr_keys) * args.data_frac))]
    tr = [d for kk in tr_keys for d in groups[kk]]
    va = [d for kk in val_keys for d in groups[kk]]
    print(f"split={args.split}: train {len(tr)} / val {len(va)}"
          f"(组 {len(keys)-n_val}/{n_val})")

    traj_n = collections.Counter(d["dir"] for d in tr)
    dq = tr[0]["q"].shape[0]
    dc = tr[0]["c"].shape[1]
    model = Energy(dq, dc).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    _run_epochs(args, model, opt, tr, va, traj_n, dev)
    return



def _run_epochs(args, model, opt, tr, va, traj_n, dev):
    def classes(d):
        n = d["c"].shape[0]
        out = []
        singles = [(f"s{i}", (i,)) for i in range(n)]
        pairs = [(f"{a}_{b}", (a, b)) for a, b in itertools.combinations(range(n), 2)]
        out.append((1, singles))
        out.append((2, pairs))
        if d["slate"] == 6:
            quads = [("q" + "_".join(map(str, S)), S)
                     for S in itertools.combinations(range(n), 4)]
            out.append((4, quads))
        return out

    def state_loss(d):
        q, c, p = d["q"].to(dev), d["c"].to(dev), d["p"].to(dev)
        loss = q.new_zeros(())
        cnt = 0
        for budget, cls in classes(d):
            names = [nm for nm, _ in cls if nm in d["oks"]]
            if len(names) < 2:
                continue
            t = torch.tensor([float(d["oks"][nm]) for nm in names], device=dev)
            if t.sum() == 0 or t.sum() == len(t):
                continue
            subsets = [S for nm, S in cls if nm in d["oks"]]
            e = model(q, c, p, budget, subsets)
            loss = loss - (torch.log_softmax(e, 0) * (t / t.sum())).sum()
            cnt += 1
        return (loss / cnt, cnt) if cnt else (None, 0)

    def evaluate(ds):
        res = {b: {"sel": [], "orc": [], "rnd": []} for b in (1, 2, 4)}
        rec_ok = []
        model.eval()
        with torch.no_grad():
            for d in ds:
                q, c, p = d["q"].to(dev), d["c"].to(dev), d["p"].to(dev)
                if "recency" in d["oks"]:
                    rec_ok.append(d["oks"]["recency"])
                for budget, cls in classes(d):
                    names = [nm for nm, _ in cls if nm in d["oks"]]
                    if not names:
                        continue
                    subsets = [S for nm, S in cls if nm in d["oks"]]
                    e = model(q, c, p, budget, subsets)
                    pick = names[int(e.argmax())]
                    oks = [d["oks"][nm] for nm in names]
                    res[budget]["sel"].append(d["oks"][pick])
                    res[budget]["orc"].append(max(oks))
                    res[budget]["rnd"].append(sum(oks) / len(oks))
        model.train()
        line = " | ".join(
            f"B={b}: sel {np.mean(v['sel']):.3f} orc {np.mean(v['orc']):.3f} "
            f"rnd {np.mean(v['rnd']):.3f}" for b, v in res.items() if v["sel"])
        return line + f" | recency {np.mean(rec_ok):.3f}(n={len(rec_ok)})"

    best_b2 = (0.0, -1, "")
    for ep in range(args.epochs):
        random.Random(ep).shuffle(tr)
        tot, seen = 0.0, 0
        opt.zero_grad()
        for i, d in enumerate(tr):
            ls, cnt = state_loss(d)
            if ls is None:
                continue
            (ls / traj_n[d["dir"]] * 4).backward()
            tot += float(ls)
            seen += 1
            if (i + 1) % 16 == 0:
                opt.step()
                opt.zero_grad()
        opt.step()
        print(f"ep{ep}  loss {tot/max(seen,1):.4f}  ({seen} 有效状态)")
        vline = evaluate(va)
        print("  val:", vline, flush=True)
        m = re.search(r"B=2: sel ([0-9.]+)", vline)
        if m and float(m.group(1)) > best_b2[0]:
            best_b2 = (float(m.group(1)), ep, vline)
        if ep % 5 == 4:
            print("  train:", evaluate(tr[:300]), flush=True)
    print(f"BEST ep{best_b2[1]}  {best_b2[2]}")


if __name__ == "__main__":
    main()
