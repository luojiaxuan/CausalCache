# note (luojiaxuan): #26 归约。判定式:
#   记忆可读性增益(arm) = [acc_arm(r2) − acc_frozen(r2)] − [acc_arm(b0) − acc_frozen(b0)]
# 显著性:同态配对的差中差,10000 次 bootstrap 给 95% CI。
import json, random
def load(p):
    d = {}
    for l in open(p):
        l = l.strip()
        if l:
            r = json.loads(l); d[r["dp_id"]] = r
    return d
arms = {"frozen": load("/data/g26/eval_frozen.jsonl"),
        "ungated": load("/data/g26/eval_ungated.jsonl"),
        "full": load("/data/g26/eval_full.jsonl")}
common = set(arms["frozen"]) & set(arms["ungated"]) & set(arms["full"])
ids = sorted(common)
print(f"三臂共同留出 hopeless 态:{len(ids)}\n")
print(f"{'臂':<10}{'B=0':>8}{'recent-2':>10}{'ΔB0':>8}{'Δr2':>8}{'记忆可读性增益':>14}")
fz = arms["frozen"]
rng = random.Random(0)
for name in ("frozen", "ungated", "full"):
    a = arms[name]
    b0 = sum(a[i]["b0_correct"] for i in ids) / len(ids)
    r2 = sum(a[i]["recent2_correct"] for i in ids) / len(ids)
    if name == "frozen":
        print(f"{name:<10}{100*b0:>7.1f}%{100*r2:>9.1f}%{'—':>8}{'—':>8}{'—':>14}")
        continue
    db0 = b0 - sum(fz[i]["b0_correct"] for i in ids) / len(ids)
    dr2 = r2 - sum(fz[i]["recent2_correct"] for i in ids) / len(ids)
    d = [ (int(a[i]["recent2_correct"]) - int(fz[i]["recent2_correct"]))
        - (int(a[i]["b0_correct"]) - int(fz[i]["b0_correct"])) for i in ids]
    boots = []
    for _ in range(10000):
        s = [d[rng.randrange(len(d))] for _ in range(len(d))]
        boots.append(sum(s) / len(s))
    boots.sort()
    lo, hi = boots[249], boots[9749]
    gain = sum(d) / len(d)
    print(f"{name:<10}{100*b0:>7.1f}%{100*r2:>9.1f}%{100*db0:>+7.1f}%{100*dr2:>+7.1f}%"
          f"{100*gain:>+9.1f}pp [CI {100*lo:+.1f},{100*hi:+.1f}]")
print("\n读法:增益 CI 下界 >0 才算'适配把 recent-2 里已有的信息解锁了';")
print("      ΔB0 大 ≈ 任务 SFT 效应(与记忆无关),两臂应大致相同。")
