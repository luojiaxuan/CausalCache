# note (luojiaxuan): 对已保存的 MemGUI-Bench 轨迹离线重判(judge 不可用时先跑 rollout,
# 之后用论文原配 judge 补判)。逐任务调用 runtime 的 evaluate_memgui_trajectory,
# 重写 result.txt 与 rejudge.jsonl;需保留各任务目录下的 screenshots/。
# 用法: uv run python memgui_rejudge.py <log_file_root> [--agent-name GUIOWL15AgentMCP] [--only-errors]
import argparse, glob, json, os, sys, time
sys.path.insert(0, "/data01/jaxan/memgui/src")
from mobile_world.runtime.utils.memgui_eval import evaluate_memgui_trajectory

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--agent-name", default="GUIOWL15AgentMCP")
ap.add_argument("--only-errors", action="store_true", help="只重判 result.txt 含 MemGUI-Eval error 的任务")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
args = ap.parse_args()
out = open(os.path.join(args.root, f"rejudge_{args.shard}.jsonl"), "a")
n = k = 0
for i, d in enumerate(sorted(glob.glob(os.path.join(args.root, "*/")))):
    task = os.path.basename(d.rstrip("/"))
    if i % args.nshards != args.shard or task.startswith("_"):
        continue
    if not os.path.exists(os.path.join(d, "traj.json")):
        continue
    rp = os.path.join(d, "result.txt")
    if args.only_errors and os.path.exists(rp) and "MemGUI-Eval" not in open(rp).read():
        continue
    score, reason = evaluate_memgui_trajectory(log_file_root=args.root, task_name=task, task_traj_dir=d.rstrip("/"),
                                               agent_name=args.agent_name, attempt_num=1)
    with open(rp, "w") as f:
        f.write(f"score: {score}\nreason: {reason}\n")
    out.write(json.dumps({"task": task, "score": score, "reason": reason[:300], "t": time.time()}) + "\n"); out.flush()
    n += 1; k += score > 0
    print(f"[{n}] {task}: {score} | {reason[:100]}", flush=True)
print(f"REJUDGE_DONE {k}/{n} = {k/max(n,1):.3f}", flush=True)
