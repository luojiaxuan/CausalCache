# note (luojiaxuan): 对账轮里"被重试的任务"与"一次跑完的任务"的成功率对比,判断池退化是否压低成功率。
import glob, os, json, sys
root = sys.argv[1]
sp = json.load(open("/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json")); tasks = sorted(set(sp["train"]) | set(sp["heldout"]), key=len, reverse=True)
bt = {}
for d in sorted(glob.glob(root + "/*/"), key=os.path.getmtime):
    n = os.path.basename(d.rstrip("/")); t = next((t for t in tasks if n == t or n.startswith(t + "-") or n.startswith(t + "_")), n); bt.setdefault(t, []).append(d)
def ok(d):
    p = os.path.join(d, "result.txt")
    if not os.path.exists(p): return False
    s = open(p, errors="ignore").read()
    return "score:" in s and float(s.split("score:")[1].split()[0]) > 0
rows = [(t, len(ds), any(ok(d) for d in ds), ok(ds[-1])) for t, ds in bt.items()]
for k in (1, 2, 3):
    sub = [r for r in rows if (r[1] == k if k < 3 else r[1] >= 3)]
    print(f"attempts{'>=' if k == 3 else '='}{k}: tasks {len(sub)}, any_success {sum(r[2] for r in sub)}, last_success {sum(r[3] for r in sub)}")
big = max(bt, key=lambda t: len(bt[t])); print("dir names sample:", [os.path.basename(d.rstrip('/')) for d in bt[big]][:4])
