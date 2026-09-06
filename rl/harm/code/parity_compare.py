# note (luojiaxuan): 逐任务比较 8-24 对账运行(traj_full)与本次官方默认协议各轮:成功翻转、重试次数、按应用聚合。
import glob, os, json, re, sys, collections
sp = json.load(open("/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json")); tasks = sorted(set(sp["train"]) | set(sp["heldout"]), key=len, reverse=True)
def load(root):
    bt = {}
    for d in sorted(glob.glob(root + "/*/"), key=os.path.getmtime):
        n = os.path.basename(d.rstrip("/")); t = next((t for t in tasks if n == t or n.startswith(t + "-") or n.startswith(t + "_")), n); bt.setdefault(t, []).append(d)
    def ok(d):
        p = os.path.join(d, "result.txt")
        if not os.path.exists(p): return None
        s = open(p, errors="ignore").read()
        return ("score:" in s and float(s.split("score:")[1].split()[0]) > 0)
    return {t: (ok(ds[-1]), len(ds), len(glob.glob(ds[-1] + "/screenshots/*.png"))) for t, ds in bt.items()}
runs = {"aug24": load("/data01/jaxan/mw/traj_full")}
for r in (1, 2, 3):
    p = f"/data01/jaxan/rl_v2/guiowl_official_default_run{r}"
    if os.path.isdir(p): runs[f"run{r}"] = load(p)
for k, v in runs.items():
    n = len(v); s = sum(1 for x in v.values() if x[0]); retried = sum(1 for x in v.values() if x[1] > 1); ex = sum(1 for x in v.values() if x[0] is False and x[2] >= 50)
    print(f"{k:6s}: 有结果 {sum(1 for x in v.values() if x[0] is not None)}/{n} 成功 {s} ({100*s/117:.1f}%) 重试任务 {retried} 50步耗尽 {ex}")
a = runs["aug24"]; b = runs["run1"]
common = [t for t in a if t in b and a[t][0] is not None and b[t][0] is not None]
both = sum(1 for t in common if a[t][0] and b[t][0]); onlya = [t for t in common if a[t][0] and not b[t][0]]; onlyb = [t for t in common if b[t][0] and not a[t][0]]
print(f"aug24 vs run1 (共 {len(common)} 题): 都成功 {both} | 仅 aug24 {len(onlya)} | 仅 run1 {len(onlyb)}")
print("  仅 aug24 成功的题(run1 重试次数, run1 步数):", [(t, b[t][1], b[t][2]) for t in onlya])
print("  仅 run1 成功:", onlyb)
def app(t):
    m = re.match(r"[A-Z]+[a-z]*", t); return m.group(0) if m else t[:6]
for k in ("aug24", "run1"):
    c = collections.Counter(); tot = collections.Counter()
    for t, x in runs[k].items():
        tot[app(t)] += 1; c[app(t)] += bool(x[0])
    print(f"  {k} 按应用 成功/总数:", {ap: f"{c[ap]}/{tot[ap]}" for ap, _ in tot.most_common(8)})
