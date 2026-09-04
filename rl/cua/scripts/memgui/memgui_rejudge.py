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
ap.add_argument("--no-resume", action="store_true", help="忽略已有 rejudge_*.jsonl 记录,重判所选任务")
args = ap.parse_args()
# note (luojiaxuan): 续传——已判过的任务(任一分片文件里有记录)跳过;单任务异常不中断整批。
done = set()
for f in ([] if args.no_resume else glob.glob(os.path.join(args.root, "rejudge_*.jsonl"))):
    for l in open(f):
        try: done.add(json.loads(l)["task"])
        except Exception: pass
out = open(os.path.join(args.root, f"rejudge_{args.shard}.jsonl"), "a")
n = k = 0
for i, d in enumerate(sorted(glob.glob(os.path.join(args.root, "*/")))):
    task = os.path.basename(d.rstrip("/"))
    # note (luojiaxuan): runner 重试时把失败尝试目录改名为 <task>_backup_<ts>,不是评测单元。
    if i % args.nshards != args.shard or task.startswith("_") or "_backup_" in task or task in done:
        continue
    if not os.path.exists(os.path.join(d, "traj.json")):
        continue
    rp = os.path.join(d, "result.txt")
    if args.only_errors and os.path.exists(rp) and "MemGUI-Eval" not in open(rp).read():
        continue
    try:
        score, reason = evaluate_memgui_trajectory(log_file_root=args.root, task_name=task, task_traj_dir=d.rstrip("/"),
                                                   agent_name=args.agent_name, attempt_num=1)
    except Exception as exc:  # noqa: BLE001
        score, reason = 0.0, f"REJUDGE_ERROR: {exc}"
    with open(rp, "w") as f:
        f.write(f"score: {score}\nreason: {reason}\n")
    out.write(json.dumps({"task": task, "score": score, "reason": reason[:300], "t": time.time()}) + "\n"); out.flush()
    n += 1; k += score > 0
    print(f"[{n}] {task}: {score} | {reason[:100]}", flush=True)
print(f"REJUDGE_DONE {k}/{n} = {k/max(n,1):.3f}", flush=True)
