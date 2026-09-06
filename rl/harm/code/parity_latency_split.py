# note (luojiaxuan): 把每步墙钟拆成"模型时延"(predict 起 → OpenAI usage 行)与"环境时延"(usage 行 → 下一次 predict 起),按题取中位数再取全体中位数。
import glob, os, re, statistics as st, sys
from datetime import datetime
pat = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \| \w+ \| \S+:(predict:\d+|_log_openai_usage:\d+) \|")
def split(root):
    model, env = [], []
    for d in glob.glob(root + "/*/"):
        if "backup" in d: continue
        ev = []
        for p in glob.glob(d + "thread_*.log"):
            for l in open(p, errors="ignore"):
                m = pat.match(l)
                if m and ("predict" in m.group(2) and "****" in l or "_log_openai_usage" in m.group(2)):
                    ev.append((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), "start" if "predict" in m.group(2) else "end"))
        ev.sort(); ms, es = [], []
        for (t0, k0), (t1, k1) in zip(ev, ev[1:]):
            dt = (t1 - t0).total_seconds()
            if k0 == "start" and k1 == "end": ms.append(dt)
            elif k0 == "end" and k1 == "start": es.append(dt)
        if len(ms) >= 3 and len(es) >= 2: model.append(st.median(ms)); env.append(st.median(es))
    return model, env
for name, root in (("aug24", "/data01/jaxan/mw/traj_full"), ("run1", "/data01/jaxan/rl_v2/guiowl_official_default_run1"), ("run3", "/data01/jaxan/rl_v2/guiowl_official_default_run3")):
    model, env = split(root)
    if model: print(f"{name:6s} 题数 {len(model):3d}  模型时延中位 {st.median(model):5.1f}s  环境时延中位 {st.median(env):5.1f}s")
