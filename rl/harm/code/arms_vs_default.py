# note (luojiaxuan): 任务级四臂对照:官方默认 N=0(×3)、recent2(既有)、recent8、random2,同池同时段;按任务配对(bootstrap 95% CI),
# 报每臂成功率、对 N=0(三轮多数票 / 各轮)的配对差、四臂并集 = 任务级"历史图余量"上界。
import glob, os, json, random, statistics as st, sys
sp = json.load(open("/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json")); tasks = sorted(set(sp["train"]) | set(sp["heldout"]), key=len, reverse=True)
def load(root):
    bt = {}
    for d in sorted(glob.glob(root + "/*/"), key=os.path.getmtime):
        n = os.path.basename(d.rstrip("/")); t = next((t for t in tasks if n == t or n.startswith(t + "-") or n.startswith(t + "_")), n); bt.setdefault(t, []).append(d)
    out = {}
    for t, ds in bt.items():
        p = os.path.join(ds[-1], "result.txt")
        if os.path.exists(p):
            s = open(p, errors="ignore").read(); out[t] = ("score:" in s and float(s.split("score:")[1].split()[0]) > 0)
    return out
arms = {f"default_r{i}": load(f"/data01/jaxan/rl_v2/guiowl_official_default_run{i}") for i in (1, 2, 3)}
for name, root in (("recent2", "/data01/jaxan/rl_v2/guiowl_base"), ("recent8", "/data01/jaxan/rl_v2/guiowl_base_recent8_v2"), ("random2", "/data01/jaxan/rl_v2/guiowl_base_random2_v2"), ("recent0_old", "/data01/jaxan/rl_v2/guiowl_base_recent0")):
    if os.path.isdir(root): arms[name] = load(root)
common = sorted(set.intersection(*[set(v) for v in arms.values()]))
print(f"共同有结果的任务 {len(common)} / 117;各臂成功率(共同任务上):")
for k, v in arms.items(): print(f"  {k:12s} {sum(v[t] for t in common):3d}/{len(common)} = {100*sum(v[t] for t in common)/len(common):5.1f}%   (全部有结果 {len(v)},成功 {sum(v.values())})")
maj = {t: sum(arms[f"default_r{i}"][t] for i in (1, 2, 3)) >= 2 for t in common}
def paired(a, b, B=2000):
    d = [int(a[t]) - int(b[t]) for t in common]; rng = random.Random(0); n = len(d); ms = sorted(st.mean(rng.choices(d, k=n)) for _ in range(B))
    return 100*st.mean(d), 100*ms[int(.025*B)], 100*ms[int(.975*B)], sum(x > 0 for x in d), sum(x < 0 for x in d)
print("\n配对差 vs N=0 三轮多数票(pp,95% CI,胜/负题数):")
for k in ("recent2", "recent8", "random2", "default_r1", "default_r2", "default_r3"):
    if k in arms: mu, lo, hi, w, l = paired(arms[k], maj); print(f"  {k:12s} {mu:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]  胜 {w} 负 {l}")
u_def = {t for t in common if any(arms[f"default_r{i}"][t] for i in (1, 2, 3))}
u_all = {t for t in common if any(v[t] for v in arms.values())}
u_hist = {t for t in common if any(arms[k][t] for k in ("recent2", "recent8", "random2") if k in arms)}
print(f"\n并集:N=0 三轮 {len(u_def)};历史图三臂 {len(u_hist)};全部 {len(u_all)};历史图臂新增(不在 N=0 三轮并集里) {len(u_hist - u_def)}:", sorted(u_hist - u_def)[:20])
