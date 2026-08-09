# note (luojiaxuan): 数据规模曲线 —— 判别"信号不可迁移"与"数据不足",
# 这两者在 train↑/holdout平 的形态下不可分,此前我把后者排除掉的论证是错的。
# 判读:
#   holdout 随训练态数上升 → 数据受限,扩标签(winmac)是正解;
#   holdout 对数据量平 → 信号确实不可迁移,该动特征形态而不是数据量。
# 固定同一个留出集(与训练器同划分),训练子集嵌套(前 25/50/75/100%),
# 每档在 1/4/16 步/态三个检查点评一次,取最好 —— 对每档数据量都给最有利读法。
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
nh = int(len(ids) * 0.2); hold, train_all = ids[:nh], ids[nh:]
print(f"winnable {len(ids)} → 固定留出 {len(hold)};训练池 {len(train_all)}")
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

def hold_acc(m, pool, ctx):
    hit = tot = 0
    with torch.no_grad():
        for dp in hold:
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

# 参照:recency 规则(子集分 = 成员帧号之和,越近越大)在同一留出集上的成绩
class Recency:
    def frame_terms(self, f, c): return None, None
    def subset_score(self, u, pr, idx_unused): raise RuntimeError
hit = tot = 0
for dp in hold:
    cands = cache[dp]["cands"]; pos, neg = lab[dp]
    r = random.Random(hash(dp) % 9973)
    for _ in range(8):
        p, q = r.choice(pos), r.choice(neg)
        if not (set(p) <= set(cands) and set(q) <= set(cands)): continue
        tot += 1; hit += int(sum(p) > sum(q))
print(f"参照:recency 规则(零参数)留出集 = {hit/max(tot,1):.4f}\n")

sizes = [len(train_all)//4, len(train_all)//2, 3*len(train_all)//4, len(train_all)]
for arch, ctx, pool, lr in (("linear", False, "mean", 1e-3),
                            ("mlp_pair", True, "mean_max", 3e-4)):
    dim = 8192 if pool == "mean_max" else 4096
    print(f"=== {arch} ctx={ctx} {pool} lr={lr} ===")
    print(f"{'训练态数':>8} {'@1步/态':>9} {'@4步/态':>9} {'@16步/态':>10} {'最好':>7}")
    for n in sizes:
        sub = train_all[:n]
        torch.manual_seed(0)
        m = SubsetScorer(dim, arch=arch, hidden=1024, rank=128, use_context=ctx).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=lr)
        rng = random.Random(1)
        marks = {1: None, 4: None, 16: None}
        step = 0
        for target in (1, 4, 16):
            while step < target * n:
                dp = sub[step % n]
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
            marks[target] = hold_acc(m, pool, ctx)
        best = max(v for v in marks.values() if v is not None)
        print(f"{n:>8} {marks[1]:>9.4f} {marks[4]:>9.4f} {marks[16]:>10.4f} {best:>7.4f}")
    print()
print("判读:'最好'列随训练态数上升 → 数据受限,扩标签是正解;平 → 信号不可迁移,该动特征形态。")
