# note (luojiaxuan): 两个判据,分开"容量不够"与"特征没信息"。
#
# 判据 A —— **帧间可分性**。打分头只能看到每帧的池化特征;若同一 state 内
# 各帧特征几乎相同,任何头都无法把它们排序,这时换头是白费。
# 量:同 state 内两两余弦相似度。接近 1 = 特征分不开帧。
#
# 判据 B —— **极端过拟合**。取 50 个 state,狂训到收敛(2000 步)。
# 若连 50 个见过的 state 都记不住(排序准确率上不去),那**排除欠训**,
# 只剩"标签不是这些特征的函数"这一种解释。
import json, random, sys, torch
sys.path.insert(0, "/data/osworld/CausalCache/rl/code")
from causalcache_rl.subset_scorer import SubsetScorer

cache = torch.load("/data/v5abl/index_features.pt", map_location="cpu", weights_only=False)
print(f"缓存 {len(cache)} 态")

# ---- A:帧间余弦相似度 ----
sims_mean, sims_mm = [], []
for e in list(cache.values())[:400]:
    f = e["feats"].float()
    for pooling, acc in (("mean_max", sims_mm), ("mean", sims_mean)):
        x = f[:, :f.shape[1] // 2] if pooling == "mean" else f
        x = torch.nn.functional.normalize(x, dim=-1)
        g = x @ x.t()
        n = g.shape[0]
        if n > 1:
            acc.append(float(g[~torch.eye(n, dtype=bool)].mean()))
for name, a in (("mean 池化", sims_mean), ("mean_max 池化", sims_mm)):
    a.sort()
    print(f"判据A {name}:同 state 内帧间余弦相似度 中位 {a[len(a)//2]:.4f}、"
          f"p10 {a[int(.1*len(a))]:.4f}、p90 {a[int(.9*len(a))]:.4f}")
print("  →接近 1 说明各帧特征几乎相同,打分头结构上无法把它们排序")

# ---- B:50 态极端过拟合 ----
lab = {}
for line in open("/data/oracle/labels_all.jsonl"):
    line = line.strip()
    if not line: continue
    d = json.loads(line)
    subs = [(tuple(a["s"]), bool(a["c"])) for a in d.get("all", []) if len(a["s"]) == 2]
    pos = [s for s, c in subs if c]; neg = [s for s, c in subs if not c]
    if pos and neg and d["dp_id"] in cache:
        lab[d["dp_id"]] = (pos, neg)
ids = sorted(lab)[:50]
print(f"\n判据B:取 {len(ids)} 态狂训 2000 步(lr 1e-3),看能不能记住")
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
for arch, ctx, pool in (("linear", False, "mean"), ("mlp_pair", True, "mean_max")):
    dim = 8192 if pool == "mean_max" else 4096
    m = SubsetScorer(dim, arch=arch, hidden=1024, rank=128, use_context=ctx).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    rng = random.Random(0)
    for step in range(2000):
        dp = ids[step % len(ids)]
        e = cache[dp]; cands = e["cands"]
        f = e["feats"].to(dev).float(); c = e["ctx"].to(dev).float()
        if pool == "mean":
            f, c = f[:, :4096], c[:4096]
        u, pr = m.frame_terms(f, c if ctx else None)
        pos, neg = lab[dp]
        loss = torch.zeros((), device=dev)
        for _ in range(8):
            p, q = rng.choice(pos), rng.choice(neg)
            if not (set(p) <= set(cands) and set(q) <= set(cands)): continue
            mg = (m.subset_score(u, pr, [cands.index(j) for j in p])
                  - m.subset_score(u, pr, [cands.index(j) for j in q]))
            loss = loss - torch.nn.functional.logsigmoid(mg)
        (loss / 8).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    hit = tot = 0
    with torch.no_grad():
        for dp in ids:
            e = cache[dp]; cands = e["cands"]
            f = e["feats"].to(dev).float(); c = e["ctx"].to(dev).float()
            if pool == "mean":
                f, c = f[:, :4096], c[:4096]
            u, pr = m.frame_terms(f, c if ctx else None)
            pos, neg = lab[dp]
            for p in pos:
                for q in neg:
                    if not (set(p) <= set(cands) and set(q) <= set(cands)): continue
                    tot += 1
                    hit += int(float(m.subset_score(u, pr, [cands.index(j) for j in p])
                                     - m.subset_score(u, pr, [cands.index(j) for j in q])) > 0)
    print(f"  {arch:<9} ctx={ctx} {pool:<9} 50 态**全部**正负对 {tot} 组,"
          f"记住 {100*hit/max(tot,1):.1f}%")
print("  → 记不住(<90%)= 排除欠训,标签不是这些特征的函数;"
      "记得住 = 之前那轮确实只是训练量/lr 问题")
