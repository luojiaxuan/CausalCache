# note (luojiaxuan): 对账轮按任务取最后一次尝试,失败归因:环境错误 / 50 步耗尽 / 主动终止但判错 / 其他;并统计重试次数。
import glob, os, json, re, sys
root = sys.argv[1]
sp = json.load(open("/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json"))
tasks = sorted(set(sp["train"]) | set(sp["heldout"]), key=len, reverse=True)
by_task = {}
for d in sorted(glob.glob(root + "/*/"), key=os.path.getmtime):
    name = os.path.basename(d.rstrip("/")); t = next((t for t in tasks if name == t or name.startswith(t + "-") or name.startswith(t + "_")), name)
    by_task.setdefault(t, []).append(d)
cat = {"success": [], "env_error": [], "step_exhausted": [], "terminated_wrong": [], "other": []}
retries = sum(len(v) - 1 for v in by_task.values())
for t, ds in by_task.items():
    d = ds[-1]; rp = os.path.join(d, "result.txt")
    if not os.path.exists(rp): cat["other"].append((t, "no result.txt")); continue
    res = open(rp, errors="ignore").read(); score = float(res.split("score:")[1].split()[0]) if "score:" in res else 0.0
    if score > 0: cat["success"].append((t, "")); continue
    steps = len(glob.glob(d + "/screenshots/*.png"))
    log = " ".join(open(p, errors="ignore").read() for p in glob.glob(d + "/thread_*.log"))
    traj = ""
    try: traj = open(os.path.join(d, "traj.json"), errors="ignore").read()[-3000:]
    except FileNotFoundError: pass
    if re.search(r"not healthy|HTTP 500|Connection refused|device offline|adb: error", log, re.I): cat["env_error"].append((t, f"steps={steps}"))
    elif steps >= 50: cat["step_exhausted"].append((t, f"steps={steps}"))
    elif re.search(r"terminate|finish", traj, re.I): cat["terminated_wrong"].append((t, f"steps={steps}"))
    else: cat["other"].append((t, f"steps={steps}"))
n = len(by_task)
print(f"{root}: 任务数={n} 目录数={sum(len(v) for v in by_task.values())} 重试={retries}")
for k, v in cat.items(): print(f"  {k:17s} {len(v):3d}  ({100*len(v)/max(n,1):.1f}%)  步数分布: {sorted(int(x.split('=')[1]) for _, x in v if '=' in x)[:20]}")
print("  env_error:", [t for t, _ in cat["env_error"]])
print("  other:", [(t, x) for t, x in cat["other"]][:15])
json.dump(cat, open(root + "_audit.json", "w"), indent=1)
