# note (luojiaxuan): 剪枝判据之争,用现有数据直接判,不花 GPU。
# B=1 全枚举了每个 state 的**所有单帧**(覆盖率 100%),B=2 知道哪些配对正确。
# 问:**正确的配对,其成员帧单独用时对不对?**
#   若多数正确配对里至少一帧单独也对 → 单帧排序(方案 a)能找到它们,剪枝安全;
#   若多数正确配对由"单独都错"的帧组成 → **任何单帧排序都会漏掉它们**,
#     那 (a) 和 (b) 都不可靠,剪枝本身才是问题。
import collections, glob, json

def load(b):
    d = {}
    for f in glob.glob(f"/data/oracle/bcurve_b{b}_sh*.jsonl") + \
             glob.glob(f"/data/oracle/h01_bcurve_b{b}_*.jsonl"):
        for l in open(f):
            l = l.strip()
            if l:
                r = json.loads(l)
                if "oracle_correct" in r: d[r["dp_id"]] = r
    return d

b1, b2 = load(1), load(2)
common = set(b1) & set(b2)
cat = collections.Counter(); pairs_tot = 0
per_state = collections.Counter()
for i in common:
    single = {tuple(a["s"])[0]: a["c"] for a in b1[i]["all"] if len(a["s"]) == 1}
    good = [tuple(a["s"]) for a in b2[i]["all"] if a["c"] and len(a["s"]) == 2]
    if not good: continue
    kinds = set()
    for p in good:
        if not set(p) <= set(single): continue
        k = sum(int(single[j]) for j in p)
        cat[k] += 1; pairs_tot += 1
        kinds.add(k)
    if kinds: per_state[max(kinds)] += 1
print(f"可比 state {len(common)};其中有正确配对的 {sum(per_state.values())}\n")
print("=== 正确配对的成员帧,单独用时有几帧是对的 ===")
for k in (0, 1, 2):
    print(f"  {k} 帧单独也对:{cat[k]:5d} 个配对 = {100*cat[k]/max(pairs_tot,1):5.1f}%")
print("\n=== 按 state 看:该 state 最好的正确配对属于哪一类 ===")
lab = {0: "**两帧单独都错**(单帧排序会漏)", 1: "恰一帧单独也对", 2: "两帧单独都对"}
tot = sum(per_state.values())
for k in (2, 1, 0):
    print(f"  {lab[k]:<32} {per_state[k]:4d} 态 = {100*per_state[k]/max(tot,1):5.1f}%")
print(f"\n若'两帧单独都错'占比高 → 正确配对是**涌现**的,任何按单帧打分的剪枝")
print(f"  都会系统性漏掉它们,这是比'搭档帧选谁'更根本的问题。")
