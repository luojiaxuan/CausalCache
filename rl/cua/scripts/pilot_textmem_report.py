# note (luojiaxuan): 外审 §3.1 的判读表:每个记忆策略在每个历史 token 预算下的命中率与实测 prompt_tokens,按 (族, 对) 聚类 bootstrap;
# 并给出判据所需的两个对比:同预算下 最佳文本策略 − 最佳图像策略,以及各策略相对"不给历史"的增益。
import argparse, json, random, re, statistics

ap = argparse.ArgumentParser()
ap.add_argument("evals", nargs="+"); ap.add_argument("--boot", type=int, default=4000)
args = ap.parse_args()
rows = [json.loads(l) for f in args.evals for l in open(f)]
tag = ",".join(sorted({r["tag"] for r in rows}))
conds = sorted({c for r in rows for c in r["hit"]})
budgets = sorted({int(m.group(1)) for c in conds if (m := re.search(r"@(\d+)$", c))})
fams = {(r["family"], r["pair"]) for r in rows}

def rate(c): 
    xs = [r["hit"][c] for r in rows if c in r["hit"]]; return sum(xs) / max(1, len(xs)), len(xs)
def tok(c):
    ts = sorted(r["ptoks"][c] for r in rows if r.get("ptoks", {}).get(c)); return statistics.median(ts) if ts else None
def paired_ci(a, b):
    d = {}
    for r in rows:
        if a in r["hit"] and b in r["hit"]: d.setdefault((r["family"], r["pair"]), []).append(r["hit"][a] - r["hit"][b])
    if not d: return None
    keys = list(d); allv = [v for k in keys for v in d[k]]; rng = random.Random(0)
    bs = sorted(100 * sum(v for k in rng.choices(keys, k=len(keys)) for v in d[k]) / len(allv) for _ in range(args.boot))
    return 100 * sum(allv) / len(allv), bs[int(.025 * args.boot)], bs[int(.975 * args.boot)], len(keys)

print(f"[{tag}] n={len(rows)} checkpoints, {len(fams)} (族,对) 簇;历史 token 预算 × 记忆策略")
base = "none"
print(f"\n{'策略':22s} " + " ".join(f"{b:>13d}" for b in budgets))
groups = {}
for c in conds:
    m = re.search(r"^(.*)@(\d+)$", c)
    if m: groups.setdefault(m.group(1), {})[int(m.group(2))] = c
for g in sorted(groups):
    cells = []
    for b in budgets:
        c = groups[g].get(b)
        if not c: cells.append(f"{'-':>13s}"); continue
        r_, n = rate(c); cells.append(f"{r_:.3f}/{tok(c) or 0:5.0f}")
    print(f"{g:22s} " + " ".join(f"{x:>13s}" for x in cells))
for c in conds:
    if "@" not in c: r_, n = rate(c); print(f"{c:22s} {r_:.3f} (n={n})  tok={tok(c)}")

print("\n判据对比(按 (族,对) 聚类 bootstrap 95%):")
for b in budgets:
    txt = [groups[g][b] for g in groups if g.startswith(("notes", "ocr")) and b in groups[g]]
    img = [groups[g][b] for g in groups if g.startswith("img") and b in groups[g]]
    if not txt or not img: continue
    bt = max(txt, key=lambda c: rate(c)[0]); bi = max(img, key=lambda c: rate(c)[0])
    d = paired_ci(bt, bi)
    print(f"  预算 {b:>5d}: 最佳文本 {bt} {rate(bt)[0]:.3f} vs 最佳图像 {bi} {rate(bi)[0]:.3f}  差 {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}] (簇 {d[3]})")
for c in conds:
    if c in (base, "gold") or "@" not in c: continue
    d = paired_ci(c, base)
    if d: print(f"  {c:20s} − {base}: {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]")
