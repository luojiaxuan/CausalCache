# note (luojiaxuan): 对账差距的"主机负载"候选:比较 8-24 运行与本次三轮的每步墙钟(截图文件 mtime 间隔的中位数)与每题总时长。
import glob, os, statistics as st, sys
def per_step(root):
    steps, totals = [], []
    for d in glob.glob(root + "/*/"):
        if "backup" in d: continue
        ts = sorted(os.path.getmtime(p) for p in glob.glob(d + "screenshots/*.png"))
        if len(ts) >= 3:
            gaps = [b - a for a, b in zip(ts, ts[1:])]; steps.append(st.median(gaps)); totals.append(ts[-1] - ts[0])
    return steps, totals
for name, root in (("aug24", "/data01/jaxan/mw/traj_full"), ("run1", "/data01/jaxan/rl_v2/guiowl_official_default_run1"), ("run2", "/data01/jaxan/rl_v2/guiowl_official_default_run2"), ("run3", "/data01/jaxan/rl_v2/guiowl_official_default_run3")):
    steps, totals = per_step(root)
    if steps: print(f"{name:6s} 题数 {len(steps):3d}  每步中位数 {st.median(steps):6.1f}s (四分位 {sorted(steps)[len(steps)//4]:5.1f}–{sorted(steps)[3*len(steps)//4]:5.1f})  每题中位时长 {st.median(totals)/60:5.1f} min")
