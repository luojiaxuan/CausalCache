# note (luojiaxuan): Venus-2 探针 v2 汇总:各臂成功率(117 全集与 heldout-39 分报)、对 recency-2 臂的
# wins/losses/ties、死循环率(末 8 步动作 ≤2 种)、撞上限率(≥50 步)、成功中位步数。用法:
# venus_tally2.py <root> <arm1> <arm2> ... ;基线臂名固定 recent2。
import glob, json, os, statistics as st, sys
root = sys.argv[1]; arms = sys.argv[2:]
split = json.load(open("/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json"))
held = set(split["heldout"])
def load(arm):
    out = {}
    for d in glob.glob(os.path.join(root, arm, "*/")):
        t = os.path.basename(d.rstrip("/")); rp = os.path.join(d, "result.txt"); tp = os.path.join(d, "traj.json")
        if not os.path.exists(rp): continue
        sc = float(open(rp).read().split("score:")[1].split()[0])
        tr = []
        if os.path.exists(tp):
            tr = (json.load(open(tp)).get("0") or {}).get("traj") or []
        sig = [json.dumps(x.get("action"), sort_keys=True) for x in tr[-8:]]
        out[t] = {"s": sc > 0, "n": len(tr), "loop": len(tr) >= 8 and len(set(sig)) <= 2, "cap": len(tr) >= 50}
    return out
R = {a: load(a) for a in arms}
base = R.get("recent2", {})
for a in arms:
    r = R[a]
    for name, sub in (("all", set(r)), ("heldout39", set(r) & held)):
        rows = [r[t] for t in sub]
        if not rows: continue
        k = sum(x["s"] for x in rows); n = len(rows)
        succ_steps = sorted(x["n"] for x in rows if x["s"])
        line = (f"{a:10s} {name:9s} success={k}/{n}={k/n*100:.1f}% loop={sum(x['loop'] for x in rows)/n*100:.0f}% "
                f"cap={sum(x['cap'] for x in rows)/n*100:.0f}% med_steps_succ={st.median(succ_steps) if succ_steps else '-'}")
        if a != "recent2" and base:
            common = sub & set(base)
            w = sum(r[t]["s"] and not base[t]["s"] for t in common); l = sum(base[t]["s"] and not r[t]["s"] for t in common)
            line += f" | vs recent2 (n={len(common)}): wins={w} losses={l} ties={len(common)-w-l}"
        print(line)
