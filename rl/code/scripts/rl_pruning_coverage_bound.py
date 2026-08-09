# note (luojiaxuan): 把"剪枝低估了多少 oracle"从担忧变成一个硬上界。
#
# 关键:**oracle 已经为真的 state,再多覆盖也不会变**。只有同时满足
#   ① oracle_correct = False,且 ② 枚举池 < 完整 C(候选数,B)
# 的 state 才**可能**因为漏枚举而被判错。这类 state 的占比就是 oracle 能
# 上涨的**硬上限** —— 即便它们全部翻对(极端假设)也不会更高。
import collections, glob, json, math

def load(pat):
    d = {}
    for f in glob.glob(pat[0]) + glob.glob(pat[1]):
        for l in open(f):
            l = l.strip()
            if l:
                r = json.loads(l)
                if "oracle_correct" in r:
                    d[r["dp_id"]] = r
    return d

data = {b: load((f"/data/oracle/bcurve_b{b}_sh*.jsonl",
                 f"/data/oracle/h01_bcurve_b{b}_*.jsonl")) for b in (1, 2, 4)}
common = set(data[1]) & set(data[2]) & set(data[4])
print(f"三档共同、且都有 oracle 的困难态:{len(common)}\n")
print(f"{'B':>3} {'oracle 实测':>11} {'完整覆盖态':>11} {'欠覆盖态':>9} "
      f"{'其中 oracle 已为真':>17} {'可能翻对(硬上限)':>18} {'oracle 上限':>11}")
for b in (1, 2, 4):
    n = len(common)
    full_cov = under = under_true = risky = 0
    orc = 0
    for i in common:
        r = data[b][i]
        pool = sum(1 for a in r.get("all", []) if a.get("e"))
        space = math.comb(r["n_candidates"], r.get("eff_budget", b))
        ok = r["oracle_correct"]; orc += ok
        if pool >= space:
            full_cov += 1
        else:
            under += 1
            if ok: under_true += 1
            else: risky += 1
    print(f"{b:>3} {100*orc/n:>10.1f}% {100*full_cov/n:>10.1f}% {100*under/n:>8.1f}% "
          f"{100*under_true/n:>16.1f}% {risky:>13}({100*risky/n:.1f}%) "
          f"{100*(orc+risky)/n:>10.1f}%")
print("\n读法:'可能翻对'= oracle 为假 **且** 池未覆盖全空间的态数,"
      "\n      即便它们**全部**因漏枚举而误判(极端上限),oracle 也只能涨到最后一列。")
