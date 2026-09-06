# note (luojiaxuan): 官方默认协议 ×3 对账的按任务去重统计 + 环境错误计数。
import glob, os, json, re, sys
sp = json.load(open("/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json"))
tasks = sorted(set(sp["train"]) | set(sp["heldout"]), key=len, reverse=True)
allok = []
for r in (1, 2, 3):
    root = f"/data01/jaxan/rl_v2/guiowl_official_default_run{r}"
    if not os.path.isdir(root): continue
    ok = set(); seen = set()
    for rp in glob.glob(root + "/*/result.txt"):
        d = os.path.basename(os.path.dirname(rp)); t = next((t for t in tasks if d == t or d.startswith(t + "-") or d.startswith(t + "_")), d); seen.add(t)
        try:
            if float(open(rp).read().split("score:")[1].split()[0]) > 0: ok.add(t)
        except Exception:
            pass
    log = open(root + ".log", errors="ignore").read(); fin = re.search(r"Final: (\d+) tasks with results, (\d+) with no results", log)
    allok.append(ok)
    print(f"  run{r}: 任务出结果 {len(seen)}/117 成功 {len(ok)} ({100*len(ok)/117:.1f}%) | unhealthy={log.count('not healthy')} http500={log.count('HTTP 500')} | {fin.group(0) if fin else '进行中'}")
if len(allok) >= 2:
    u = set().union(*allok); i = set.intersection(*allok)
    print(f"  跨轮:并集 {len(u)} 交集 {len(i)} 均值 {sum(len(o) for o in allok)/len(allok):.1f}/117 = {100*sum(len(o) for o in allok)/len(allok)/117:.1f}%")
