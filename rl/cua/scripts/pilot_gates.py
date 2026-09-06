# note (luojiaxuan): B-pilot 三档判定的计算(设计稿 §5):全样本配对 ΔV(source − ctrl(同龄无证据轮)/ source − recency2 / source − text_only),
# 按基础对(pair)聚类的 bootstrap 95% 区间;孪生翻转率 P(source→本版答案 ∧ swap→另一版答案);gold_text 有效性;leak 分层。
# 条件名按后端取默认:owl = src_keep / ctrl_keep / swap_keep,venus = src_at_turn / ctrl_at_turn / swap_at_turn;可用 --source/--ctrl/--swap 覆盖。
# 用法:pilot_gates.py <eval.jsonl> [--source ...] [--ctrl ...] [--swap ...] [--boot 4000]
import argparse, json, random, sys
ap = argparse.ArgumentParser()
ap.add_argument("evals", nargs="+"); ap.add_argument("--source", default=""); ap.add_argument("--ctrl", default=""); ap.add_argument("--swap", default="")
ap.add_argument("--boot", type=int, default=4000); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rows = [json.loads(l) for f in args.evals for l in open(f)]
tag = ",".join(sorted({r["tag"] for r in rows}))
suffix = "_at_turn" if rows and rows[0].get("backend") == "venus" else "_keep"
args.source = args.source or "src" + suffix; args.ctrl = args.ctrl or "ctrl" + suffix; args.swap = args.swap or "swap" + suffix
conds = sorted({c for r in rows for c in r["hit"]})
print(f"[{tag}] n={len(rows)} pairs={len({(r['family'], r['pair']) for r in rows})} families={sorted({r['family'] for r in rows})}")
print(f"  gold_text 有效性: {sum(r['hit'].get('gold_text', 0) for r in rows)}/{len(rows)}   leak: goal={sum(r['leak']['goal'] for r in rows)} hist={sum(r['leak']['hist'] for r in rows)}")
print("  V(条件) = 动作携带正确答案的比例:")
for c in conds:
    xs = [r["hit"][c] for r in rows if c in r["hit"]]; print(f"    {c:22s} {sum(xs)/len(xs):.3f} (n={len(xs)})")

def delta(rs, a, b): return [r["hit"][a] - r["hit"][b] for r in rs if a in r["hit"] and b in r["hit"]]

def boot_ci(rs, a, b):
    clusters = {}
    for r in rs:
        if a in r["hit"] and b in r["hit"]: clusters.setdefault((r["family"], r["pair"]), []).append(r["hit"][a] - r["hit"][b])
    keys = list(clusters); rng = random.Random(args.seed); stats = []
    for _ in range(args.boot):
        samp = [d for k in rng.choices(keys, k=len(keys)) for d in clusters[k]]; stats.append(sum(samp) / len(samp))
    stats.sort(); return stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))], len(keys)

print(f"  闸门(source = {args.source};ΔV 单位 pp;区间 = 按 pair 聚类 bootstrap 95%):")
for ctrl in (args.ctrl, "rec2", "text_only", args.swap, "irr2"):
    d = delta(rows, args.source, ctrl)
    if not d: print(f"    source − {ctrl:10s}: 无该条件"); continue
    lo, hi, k = boot_ci(rows, args.source, ctrl); m = sum(d) / len(d)
    verdict = "GO-级" if m >= 0.10 and lo > 0 else ("EXPAND-级" if m >= 0.10 else "未过线")
    print(f"    source − {ctrl:10s}: {100*m:+6.1f}  [{100*lo:+6.1f}, {100*hi:+6.1f}]  n={len(d)} pairs={k}  {verdict}")
flips = [r for r in rows if args.source in r["hit"] and args.swap in r["hit"]]
if flips:
    f = sum(r["hit"][args.source] == 1 and r["hit"][args.swap] == 1 for r in flips) / len(flips)
    print(f"  孪生翻转 P(source→本版 ∧ swap→另一版) = {f:.3f} (n={len(flips)});仅 swap→另一版 = {sum(r['hit'][args.swap] for r in flips)/len(flips):.3f}")
for lk in ("goal", "hist"):
    a = [r for r in rows if r["leak"][lk]]; b = [r for r in rows if not r["leak"][lk]]
    if a and b:
        for name, rs in (("leak", a), ("clean", b)):
            d = delta(rs, args.source, "rec2"); print(f"  按 leak.{lk} 分层 {name:5s}: source − rec2 = {100*sum(d)/max(len(d),1):+6.1f} (n={len(d)})")
