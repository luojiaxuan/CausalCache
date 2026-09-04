# note (luojiaxuan): 列出两臂中轨迹为空(未真正执行)的任务,生成重跑用的 task CSV(沿用官方列)。
import csv, json, os, sys
R = "/data01/jaxan/rl_v2/memgui"; rows = list(csv.DictReader(open("/data01/jaxan/memgui/data/memgui-tasks-all.csv")))
for arm in sys.argv[1:] or ("armA_off_hist1", "armB_off_recency_h3"):
    miss = []
    for r in rows:
        t = r["task_identifier"]; p = f"{R}/{arm}/{t}/traj.json"
        try:
            d = json.load(open(p)); ok = isinstance(d, dict) and len(d) > 0 and len(next(iter(d.values())).get("traj", [])) > 0
        except Exception:
            ok = False
        if not ok: miss.append(r)
    out = f"{R}/{arm}_missing.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(miss)
    print(f"{arm}: missing={len(miss)}/128 -> {out}")
