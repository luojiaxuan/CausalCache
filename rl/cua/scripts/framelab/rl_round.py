# note (luojiaxuan): selector-RL 回合编排。每回合:采样任务 × G 个 tag 的
# mw-eval rollout(同任务多 rollout 构成 RLOO 组),收官后扫 traj 根提取
# 回报(gmd5+tag 对账键)。sidecar trainer 守护进程独立消费。
import glob, hashlib, json, os, random, subprocess, sys, time

D = "/data01/jaxan/mw"
ROOT = "/data01/jaxan/rl_v2"
RETURNS = f"{ROOT}/returns.jsonl"
G = int(os.environ.get("RL_G", "8"))
N_TASKS = int(os.environ.get("RL_TASKS", "8"))
CONC_TAGS = int(os.environ.get("RL_CONC_TAGS", "2"))
LLM = os.environ["RL_LLM_URL"]
SEL = os.environ["RL_SEL_URL"]
ROUND0 = int(os.environ.get("RL_ROUND0", "0"))
N_ROUNDS = int(os.environ.get("RL_ROUNDS", "10000"))

split = json.load(open("/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json"))
TRAIN = sorted(split.get("train") or split.get("train_tasks"))


def run_tag(rnd, tag, tasks):
    out = f"{ROOT}/round_{rnd}/tag{tag}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ)
    env.update({"CC_HISTORY_N": "3", "CC_FRAME_POLICY": "learned",
                "CC_SELECTOR_URL": SEL, "CC_EP_TAG": f"r{rnd}t{tag}",
                "PYTHONPATH": "/data01/jaxan/pyshim"})
    hosts = subprocess.run(
        ["bash", "-c",
         "cut -f4 /data01/jaxan/sglang-omni-rl/pool_ports.tsv | sed 's|^|http://127.0.0.1:|' | paste -sd,"],
        capture_output=True, text=True).stdout.strip()
    cmd = ["timeout", os.environ.get("RL_TAG_TIMEOUT", "7200"),
           "uv", "run", "mw", "eval", "--agent_type", "gui_owl_1_5",
           "--task", ",".join(tasks), "--max_round", "50",
           "--model_name", "gui-owl", "--llm_base_url", LLM, "--api_key", "EMPTY",
           "--step_wait_time", "3", "--max-concurrency", "8",
           "--aw-host", hosts, "--log_file_root", out]
    log = open(f"{ROOT}/round_{rnd}/tag{tag}.log", "w")
    return subprocess.Popen(cmd, cwd=f"{D}/MobileWorld", env=env,
                            stdout=log, stderr=subprocess.STDOUT)


def harvest(rnd, tag):
    out = f"{ROOT}/round_{rnd}/tag{tag}"
    n = 0
    with open(RETURNS, "a") as fo:
        for d in glob.glob(os.path.join(out, "*/")):
            rp, tp = os.path.join(d, "result.txt"), os.path.join(d, "traj.json")
            if not (os.path.exists(rp) and os.path.exists(tp)):
                continue
            try:
                score = float(open(rp).read().split("score:")[1].split()[0])
                goal = next(iter(json.load(open(tp)).values())).get("traj")[0].get("task_goal", "")
            except Exception:  # noqa: BLE001
                continue
            fo.write(json.dumps({
                "tag": f"r{rnd}t{tag}", "task": os.path.basename(d.rstrip("/")),
                "gmd5": hashlib.md5(goal.encode()).hexdigest()[:16],
                "score": 1.0 if score > 0 else 0.0, "t": time.time()}) + "\n")
            n += 1
    return n


def main():
    os.makedirs(ROOT, exist_ok=True)
    for rnd in range(ROUND0, ROUND0 + N_ROUNDS):
        tasks = random.Random(20260902 + rnd).sample(TRAIN, N_TASKS)
        print(f"[round {rnd}] tasks={tasks}", flush=True)
        pend = list(range(G))
        running = []
        while pend or running:
            while pend and len(running) < CONC_TAGS:
                tag = pend.pop(0)
                running.append((tag, run_tag(rnd, tag, tasks)))
                print(f"[round {rnd}] tag{tag} 起", flush=True)
            for tag, pr in list(running):
                if pr.poll() is not None:
                    running.remove((tag, pr))
                    n = harvest(rnd, tag)
                    print(f"[round {rnd}] tag{tag} 收 {n} 回报(exit={pr.returncode})",
                          flush=True)
            time.sleep(20)
        print(f"ROUND_DONE {rnd}", flush=True)
    print("RL_LOOP_DONE", flush=True)


if __name__ == "__main__":
    main()
