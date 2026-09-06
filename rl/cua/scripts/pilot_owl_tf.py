# note (luojiaxuan): 教师强制前缀:沿 Venus 已跑出的 Mail 轨迹(同一串屏幕),让 GUI-Owl 在每一屏上按自己的部署协议(最近两帧 + 之前各步的一行 conclusion)
# 写出自己的回复,但执行的动作按 Venus 的原路走。产物是 GUI-Owl 格式的 traj.json(+ 指向 Venus 截图的软链),供 checkpoint 矩阵在
# "一行摘要的 agent"上评估:它自己写的摘要有没有把值带到决策步,带不到时源帧能不能救回。
import argparse, glob, importlib.util, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True, help="Venus Mail 规格 jsonl(dir/step/…)")
ap.add_argument("--base-url", required=True); ap.add_argument("--model", default="gui-owl")
ap.add_argument("--out-root", required=True); ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()
go = load("/data01/jaxan/guiowl_oracle.py", "go")
sys.argv = ["decode_ctx.py", "--base-url", args.base_url, "--tag", "x", "--out", "/dev/null", "--specs", "rec0", "--roots", "/nonexistent"]
dc = load("/data01/jaxan/decode_ctx.py", "dc")

def shots_of(d):
    return sorted(glob.glob(os.path.join(d, "screenshots", "*.png")), key=lambda p: int(re.search(r"-(\d+)\.png$", p).group(1)))

def force(sp):
    src = sp["dir"]; task = sp["task"]; k = sp["step"]; out = os.path.join(args.out_root, task)
    if os.path.exists(os.path.join(out, "result.txt")): return
    os.makedirs(out, exist_ok=True)
    link = os.path.join(out, "screenshots")
    if not os.path.exists(link): os.symlink(os.path.abspath(os.path.join(src, "screenshots")), link)
    traj_v = list(json.load(open(os.path.join(src, "traj.json"))).values())[0]["traj"]; goal = traj_v[0].get("task_goal", "")
    shots = shots_of(src); preds = {}; concls = []; rows = []
    for j in range(1, k):
        st = {"dir": out, "task": task, "goal": goal, "step": j, "shots": shots, "concls": list(concls), "preds": dict(preds), "target": ""}
        dc._resp_cache[out] = dict(preds)
        msgs = dc.messages_deploy(st, 2)
        txt = go.post(args.base_url, {"model": args.model, "temperature": 0.0, "max_tokens": 256, "messages": msgs})["choices"][0]["message"]["content"] or ""
        preds[j] = txt; concls.append(go.extract_conclusion(txt)); rows.append({"step": j, "prediction": txt, "task_goal": goal, "forced_from": src})
    dc._resp_cache.pop(out, None)
    json.dump({"0": {"traj": rows}}, open(os.path.join(out, "traj.json"), "w"), ensure_ascii=False)
    open(os.path.join(out, "result.txt"), "w").write("score: 0.0\nreason: teacher-forced prefix along the Venus path; no closed-loop outcome\n")
    print(f"{task} forced {k-1} steps", flush=True)

specs = [json.loads(l) for l in open(args.spec)]
with ThreadPoolExecutor(args.workers) as ex: list(ex.map(force, specs))
# 规格:同一批 checkpoint,目录换成教师强制产物(孪生 / 无关目录同样换)
def remap(d): return os.path.join(args.out_root, os.path.basename(d.rstrip("/"))) if d else d
with open(os.path.join(args.out_root, "specs_owl_tf.jsonl"), "w") as f:
    for sp in specs:
        r = dict(sp); r["dir"] = remap(sp["dir"])
        for key in ("swap_frames_dir", "irrelevant_dir"):
            if sp.get(key) and os.path.exists(os.path.join(remap(sp[key]), "traj.json")): r[key] = remap(sp[key])
            elif key in r: del r[key]
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("specs ->", os.path.join(args.out_root, "specs_owl_tf.jsonl"))
