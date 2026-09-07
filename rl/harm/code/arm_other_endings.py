# note (luojiaxuan): 看 recent8 臂里被归入"其它"的 19 个结束方式:最后一个动作是什么、步数、日志尾有没有异常。
import glob, os, json, re
sp = json.load(open("/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json")); tasks = sorted(set(sp["train"]) | set(sp["heldout"]), key=len, reverse=True)
root = "/data01/jaxan/rl_v2/guiowl_base_recent8_v2"; bt = {}
for d in sorted(glob.glob(root + "/*/"), key=os.path.getmtime):
    n = os.path.basename(d.rstrip("/")); t = next((t for t in tasks if n == t or n.startswith(t + "-") or n.startswith(t + "_")), n); bt.setdefault(t, []).append(d)
import collections; kinds = collections.Counter()
for t, ds in bt.items():
    d = ds[-1]; rp = os.path.join(d, "result.txt")
    if not os.path.exists(rp): kinds["no result.txt"] += 1; continue
    s = open(rp, errors="ignore").read(); ok = "score:" in s and float(s.split("score:")[1].split()[0]) > 0
    n = len(glob.glob(d + "screenshots/*.png"))
    if ok or n >= 50: continue
    try: traj = json.load(open(os.path.join(d, "traj.json")))
    except Exception as e: kinds[f"traj.json unreadable"] += 1; continue
    txt = json.dumps(traj)[-4000:].lower()
    if "terminate" in txt: continue
    # 最后一个动作类型
    acts = re.findall(r'"action_type":\s*"([a-z_]+)"', json.dumps(traj)); last = acts[-1] if acts else "none"
    log = " ".join(open(p, errors="ignore").read()[-3000:] for p in glob.glob(d + "thread_*.log"))
    err = "exception" if re.search(r"traceback|exception|error", log, re.I) else ""
    kinds[f"last={last} steps~{n//10*10} {err}"] += 1
for k, v in kinds.most_common(): print(f"  {v:3d}  {k}")
