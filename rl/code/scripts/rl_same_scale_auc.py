# note (luojiaxuan): 把所有臂放到**同一把尺**上 —— 帧级 hi/lo 承载率配对 AUC。
# 此前我把"训练头的子集对排序 0.583"与"注意力的帧级 AUC 0.597"并排比较,
# 那是两个不同指标,比较无效。这里统一重测。
#
# 关键新增:**recency 单独一条臂**,以及 **neural + recency 联合**。
# 符号基线显示"越近越好"就有 64.6%,而我们的打分头**从未把 recency 作为
# 显式输入** —— 池化 hidden state 里位置信息很弱。所以真正要问的是:
#   **神经特征在 recency 之上还能加什么?** 加不了 → selector 该做成规则。
import json, math, random, sys, torch
sys.path.insert(0, "/data/osworld/CausalCache/rl/code")
from causalcache_rl.subset_scorer import SubsetScorer

cache = torch.load("/data/v5abl/index_features.pt", map_location="cpu", weights_only=False)
lab, carrier = {}, {}
for line in open("/data/oracle/labels_all.jsonl"):
    line = line.strip()
    if not line: continue
    d = json.loads(line); dp = d["dp_id"]
    subs = [(tuple(a["s"]), bool(a["c"])) for a in d.get("all", []) if len(a["s"]) == 2]
    pos = [s for s, c in subs if c]; neg = [s for s, c in subs if not c]
    if not (pos and neg and dp in cache): continue
    lab[dp] = (pos, neg)
    cr = {}
    for sset, c in subs:
        for j in sset:
            h, t = cr.get(j, (0, 0)); cr[j] = (h + int(c), t + 1)
    rate = {j: h / t for j, (h, t) in cr.items() if t >= 2}
    if len(rate) >= 2:
        hi = max(rate, key=rate.get); lo = min(rate, key=rate.get)
        if rate[hi] - rate[lo] >= 0.2:
            carrier[dp] = (hi, lo)
ids = sorted(lab); random.Random(20260809).shuffle(ids)
nh = int(len(ids) * 0.2); hold, train = ids[:nh], ids[nh:]
hold_c = [d for d in hold if d in carrier]
print(f"winnable {len(ids)};留出 {len(hold)},其中可判帧对 {len(hold_c)}")

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
D = 4096

def frame_auc(score_fn):
    """帧级 AUC:高承载帧的分数是否高于低承载帧。与符号/注意力诊断同构。"""
    hit = tot = 0
    for dp in hold_c:
        hi, lo = carrier[dp]; cands = cache[dp]["cands"]
        if hi not in cands or lo not in cands: continue
        u = score_fn(dp)
        a, b = float(u[cands.index(hi)]), float(u[cands.index(lo)])
        if a == b: continue
        tot += 1; hit += int(a > b)
    return hit / max(tot, 1), tot

# 臂 1:recency(越近越好)—— 纯规则,无参数
r_auc, r_n = frame_auc(lambda dp: torch.tensor(
    [float(j) for j in cache[dp]["cands"]]))

# 臂 2/3:神经头;臂 3 额外把 recency 作为显式输入拼进特征
def train_head(with_recency, steps_per_state=16, lr=1e-3):
    dim = D + (1 if with_recency else 0)
    torch.manual_seed(0)
    m = SubsetScorer(dim, arch="linear").to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    rng = random.Random(1)
    def feats_of(dp):
        e = cache[dp]; f = e["feats"].to(dev).float()[:, :D]
        if with_recency:
            n = f.shape[0]
            # 归一化位置:1 = 最近。不归一化则尺度随历史长度漂移
            pos = torch.tensor([[(i + 1) / n] for i in range(n)], device=dev)
            f = torch.cat([f, pos], dim=-1)
        return f
    for step in range(steps_per_state * len(train)):
        dp = train[step % len(train)]
        cands = cache[dp]["cands"]; f = feats_of(dp)
        u, pr = m.frame_terms(f, None)
        pos_s, neg_s = lab[dp]
        loss = torch.zeros((), device=dev); k = 0
        for _ in range(8):
            p, q = rng.choice(pos_s), rng.choice(neg_s)
            if not (set(p) <= set(cands) and set(q) <= set(cands)): continue
            loss = loss - torch.nn.functional.logsigmoid(
                m.subset_score(u, pr, [cands.index(j) for j in p])
                - m.subset_score(u, pr, [cands.index(j) for j in q])); k += 1
        if k:
            (loss / k).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        return frame_auc(lambda dp: m.frame_terms(feats_of(dp), None)[0])

n_auc, n_n = train_head(False)
nr_auc, nr_n = train_head(True)

print(f"\n=== 同一把尺:留出集帧级 hi/lo 承载率 AUC ===")
print(f"{'臂':<34}{'AUC':>8}{'有效对':>8}")
print(f"{'recency(越近越好,零参数规则)':<30}{100*r_auc:>7.1f}%{r_n:>8}")
print(f"{'神经特征(不含 recency)':<32}{100*n_auc:>7.1f}%{n_n:>8}")
print(f"{'神经特征 + 显式 recency':<32}{100*nr_auc:>7.1f}%{nr_n:>8}")
def p2(k, m_):
    t = sum(math.comb(m_, i) for i in range(min(k, m_ - k) + 1)) / (2 ** m_)
    return min(1.0, 2 * t)
kr, kn = round(r_auc * r_n), round(nr_auc * nr_n)
print(f"\n神经+recency 相对 recency 单独:{100*(nr_auc-r_auc):+.1f}pp")
print("  → 若 ≈0,说明神经特征在'越近越好'之上加不了东西,"
      "\n    可迁移信号的全部内容就是 recency,而 recent-B 已经把它用满了。")
