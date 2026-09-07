# note (luojiaxuan): 闭环各臂的结束方式:成功 / 主动终止但判错(过早终止)/ 50 步耗尽 / 其它;步数中位数。看步级的终止塌陷在任务级变成了什么。
import glob, os, json, statistics as st, sys
sp = json.load(open("/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json")); tasks = sorted(set(sp["train"]) | set(sp["heldout"]), key=len, reverse=True)
def analyse(root):
    bt = {}
    for d in sorted(glob.glob(root + "/*/"), key=os.path.getmtime):
        n = os.path.basename(d.rstrip("/")); t = next((t for t in tasks if n == t or n.startswith(t + "-") or n.startswith(t + "_")), n); bt.setdefault(t, []).append(d)
    c = {"success": 0, "terminated_wrong": 0, "exhausted": 0, "other": 0}; steps_term = []; steps_all = []
    for t, ds in bt.items():
        d = ds[-1]; rp = os.path.join(d, "result.txt")
        if not os.path.exists(rp): c["other"] += 1; continue
        s = open(rp, errors="ignore").read(); ok = "score:" in s and float(s.split("score:")[1].split()[0]) > 0
        n = len(glob.glob(d + "screenshots/*.png")); steps_all.append(n)
        if ok: c["success"] += 1; continue
        try: last = json.load(open(os.path.join(d, "traj.json")))
        except Exception: c["other"] += 1; continue
        txt = json.dumps(last)[-4000:].lower()
        if n >= 50: c["exhausted"] += 1
        elif "terminate" in txt: c["terminated_wrong"] += 1; steps_term.append(n)
        else: c["other"] += 1
    return c, (st.median(steps_term) if steps_term else None), (st.median(steps_all) if steps_all else None), len(bt)
for name, root in (("default_r1", "/data01/jaxan/rl_v2/guiowl_official_default_run1"), ("default_r2", "/data01/jaxan/rl_v2/guiowl_official_default_run2"), ("default_r3", "/data01/jaxan/rl_v2/guiowl_official_default_run3"), ("recent2(09-05)", "/data01/jaxan/rl_v2/guiowl_base"), ("recent8", "/data01/jaxan/rl_v2/guiowl_base_recent8_v2"), ("random2", "/data01/jaxan/rl_v2/guiowl_base_random2_v2")):
    if not os.path.isdir(root): continue
    c, mt, ma, n = analyse(root)
    print(f"{name:15s} n={n:3d}  成功 {c['success']:3d}  过早终止 {c['terminated_wrong']:3d}(中位步 {mt})  50步耗尽 {c['exhausted']:3d}  其它 {c['other']:3d}  全部中位步 {ma}")
