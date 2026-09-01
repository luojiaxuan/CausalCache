# note (luojiaxuan): 序贯 vs 联合的离线判决。用审计标签(单帧+双帧真值)
# 模拟三种选择规则在**完美价值估计**下的表现,全部以 recency 为基线:
#   oracle    = argmax_{i<j} u_ij(联合打分的天花板)
#   greedy-seq= c1 = argmax_i u_i;c2 = argmax_j u_{c1,j}(序贯贪心的天花板)
#   additive  = argmax_{i<j} u_i + u_j(纯可加基线)
# FOG 用聚合比(Σ超额/Σoracle超额),不用逐状态比值均值(外审:分母近零病态)。
import glob, itertools, json, statistics as st

K = 6
recs = []
for f in sorted(glob.glob("/data01/jaxan/labels_audit_shard*.jsonl")):
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if (r.get("recency") is not None and len(r.get("pairs") or {}) == 15
                and len(r.get("singles") or {}) == K
                and all(v is not None for v in r["pairs"].values())
                and all(v is not None for v in r["singles"].values())):
            recs.append(r)

orc, seq, add, rnd = [], [], [], []
seq_hit = 0
for r in recs:
    base = r["recency"]
    P, S = r["pairs"], r["singles"]
    okey = max(P, key=P.get)
    orc.append(P[okey] - base)
    c1 = max(range(K), key=lambda i: S[str(i)])
    c2 = max((j for j in range(K) if j != c1),
             key=lambda j: P[f"{min(c1,j)}_{max(c1,j)}"])
    skey = f"{min(c1,c2)}_{max(c1,c2)}"
    seq.append(P[skey] - base)
    if skey == okey:
        seq_hit += 1
    akey = max(itertools.combinations(range(K), 2),
               key=lambda ab: S[str(ab[0])] + S[str(ab[1])])
    add.append(P[f"{akey[0]}_{akey[1]}"] - base)
    rnd.append(st.mean(P.values()) - base)

n = len(recs)
so, ss, sa, sr = sum(orc), sum(seq), sum(add), sum(rnd)
print(f"状态数 {n}")
print(f"  oracle 联合     : 均值 {so/n:+.4f}   FOG 100%")
print(f"  greedy 序贯     : 均值 {ss/n:+.4f}   FOG {ss/so:.0%}   top-pair 命中 {seq_hit/n:.0%}")
print(f"  纯可加          : 均值 {sa/n:+.4f}   FOG {sa/so:.0%}")
print(f"  随机双帧        : 均值 {sr/n:+.4f}   FOG {sr/so:.0%}")
