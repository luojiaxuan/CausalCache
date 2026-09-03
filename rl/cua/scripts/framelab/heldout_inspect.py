import glob, json, os
S = json.load(open("/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json"))
print("=== split _meta ===", json.dumps(S["_meta"], ensure_ascii=False)[:600])
print("strict_alt keys:", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in S["strict_alt"].items()})
os.chdir("/data01/jaxan/rl_v2/anchor")
def res(label):
    o = {}
    for d in glob.glob(f"{label}/*/"):
        try:
            o[os.path.basename(d.rstrip("/"))] = int(float(open(d + "result.txt").read().split("score:")[1].split()[0]) > 0)
        except Exception:
            pass
    return o
def goal(t):
    for label in ("recency_r18", "learned_r18", "recency_base", "learned_r21"):
        try:
            tj = json.load(open(f"{label}/{t}/traj.json"))
            tr = next(iter(tj.values()))["traj"]
            return tr[0].get("task_goal", "")[:170].replace("\n", " "), len(tr)
        except Exception:
            continue
    return "?", -1
rb, r18, l18, l21 = res("recency_base"), res("recency_r18"), res("learned_r18"), res("learned_r21")
tasks = sorted(set(rb) | set(r18) | set(l18) | set(l21))
print("=== heldout-20:任务 | rec_base rec18 L18 L21 | 步数 | goal ===")
for t in tasks:
    g, n = goal(t)
    print(f"{t[:32]:32s} {rb.get(t,'-')} {r18.get(t,'-')} {l18.get(t,'-')} {l21.get(t,'-')} | {n:3d} | {g}")
