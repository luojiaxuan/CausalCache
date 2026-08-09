# note (luojiaxuan): 训练长度扫描,回答唯一还没分清的问题 ——
# 训练集拟合度上不去,到底是 ①步数不够 还是 ②信号本身不可迁移?
# 判读:
#   train↑ 且 holdout↑  → 之前纯粹是欠训,继续加训练量就行;
#   train↑ 但 holdout 平 → 学到的是**逐 state 的记忆**,没有可迁移的规则。
#     这种情况下再堆容量或再加数据都没用,要动的是**任务表述或特征**。
# 全部在缓存特征上跑,不占 8B。
import json, random, sys, torch
sys.path.insert(0, "/data/osworld/CausalCache/rl/code")
from causalcache_rl.subset_scorer import SubsetScorer

cache = torch.load("/data/v5abl/index_features.pt", map_location="cpu", weights_only=False)
lab = {}
for line in open("/data/oracle/labels_all.jsonl"):
    line = line.strip()
    if not line: continue
    d = json.loads(line)
    subs = [(tuple(a["s"]), bool(a["c"])) for a in d.get("all", []) if len(a["s"]) == 2]
    pos = [s for s, c in subs if c]; neg = [s for s, c in subs if not c]
    if pos and neg and d["dp_id"] in cache:
        lab[d["dp_id"]] = (pos, neg)
ids = sorted(lab); random.Random(20260809).shuffle(ids)
nh = int(len(ids) * 0.2)
hold, train = ids[:nh], ids[nh:]
print(f"winnable {len(ids)} → train {len(train)} / holdout {len(hold)}(与训练器同划分)")
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

def acc(m, which, pool, ctx, n=None):
    hit = tot = 0
    with torch.no_grad():
        for dp in (which if n is None else which[:n]):
            e = cache[dp]; cands = e["cands"]
            f = e["feats"].to(dev).float(); c = e["ctx"].to(dev).float()
            if pool == "mean": f, c = f[:, :4096], c[:4096]
            u, pr = m.frame_terms(f, c if ctx else None)
            pos, neg = lab[dp]
            r = random.Random(hash(dp) % 9973)
            for _ in range(8):
                p, q = r.choice(pos), r.choice(neg)
                if not (set(p) <= set(cands) and set(q) <= set(cands)): continue
                tot += 1
                hit += int(float(m.subset_score(u, pr, [cands.index(j) for j in p])
                                 - m.subset_score(u, pr, [cands.index(j) for j in q])) > 0)
    return hit / max(tot, 1)

for arch, ctx, pool, lr in (("linear", False, "mean", 1e-3),
                            ("mlp_pair", True, "mean_max", 3e-4)):
    dim = 8192 if pool == "mean_max" else 4096
    torch.manual_seed(0)
    m = SubsetScorer(dim, arch=arch, hidden=1024, rank=128, use_context=ctx).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    rng = random.Random(1)
    print(f"\n=== {arch} ctx={ctx} {pool} lr={lr} ===")
    print(f"{'每态步数':>8} {'总步数':>8} {'训练集':>8} {'留出集':>8}")
    step = 0
    for target in (1, 4, 16, 64):
        while step < target * len(train):
            dp = train[step % len(train)]
            e = cache[dp]; cands = e["cands"]
            f = e["feats"].to(dev).float(); c = e["ctx"].to(dev).float()
            if pool == "mean": f, c = f[:, :4096], c[:4096]
            u, pr = m.frame_terms(f, c if ctx else None)
            pos, neg = lab[dp]
            loss = torch.zeros((), device=dev); k = 0
            for _ in range(8):
                p, q = rng.choice(pos), rng.choice(neg)
                if not (set(p) <= set(cands) and set(q) <= set(cands)): continue
                loss = loss - torch.nn.functional.logsigmoid(
                    m.subset_score(u, pr, [cands.index(j) for j in p])
                    - m.subset_score(u, pr, [cands.index(j) for j in q])); k += 1
            if k:
                (loss / k).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
        print(f"{target:>8} {step:>8} {acc(m, train, pool, ctx, 250):>8.4f} "
              f"{acc(m, hold, pool, ctx):>8.4f}")
