# note (luojiaxuan): heldout 评测锚。每隔 ANCHOR_EVERY 个训练回合,用当前
# selector 权重在固定的 heldout 任务子集上跑一遍 learned 策略,rate 写
# heldout_curve.jsonl——回答"训练有没有在涨"的唯一正规口径(训练任务的
# 回合均值混着任务抽样噪声,不作趋势判据)。recency 基线因 executor 冻结
# 而恒定,只测一次作参照线。env 用池的后 8 台,与训练 rollout(前 16 台
# 常用)错开。
import glob, hashlib, json, os, re, subprocess, time

D = "/data01/jaxan/mw"
ROOT = "/data01/jaxan/rl_v2"
CURVE = f"{ROOT}/heldout_curve.jsonl"
EVERY = int(os.environ.get("ANCHOR_EVERY", "3"))
N_TASKS = int(os.environ.get("ANCHOR_TASKS", "20"))
LLM = os.environ["RL_LLM_URL"]
SEL = os.environ["RL_SEL_URL"]

split = json.load(open("/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json"))
import random
TASKS = sorted(random.Random(20260903).sample(sorted(split["heldout"]), N_TASKS))
HOSTS = ",".join(f"http://127.0.0.1:{p}" for p in range(6824, 6832))


def cur_round():
    try:
        lines = [l for l in open(f"{ROOT}/rounds.log") if "ROUND_DONE" in l]
        return int(lines[-1].split()[-1]) if lines else -1
    except Exception:  # noqa: BLE001
        return -1


def evaluated():
    done = set()
    if os.path.exists(CURVE):
        for l in open(CURVE):
            try:
                done.add(json.loads(l)["round"])
            except Exception:  # noqa: BLE001
                continue
    return done


def run_eval(rnd, policy, label):
    out = f"{ROOT}/anchor/{label}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ)
    env.update({"CC_HISTORY_N": "3", "CC_FRAME_POLICY": policy,
                "CC_SELECTOR_URL": SEL, "CC_EP_TAG": label,
                "PYTHONPATH": "/data01/jaxan/pyshim"})
    cmd = ["timeout", "5400", "uv", "run", "mw", "eval", "--agent_type", "gui_owl_1_5",
           "--task", ",".join(TASKS), "--max_round", "50",
           "--model_name", "gui-owl", "--llm_base_url", LLM, "--api_key", "EMPTY",
           "--step_wait_time", "3", "--max-concurrency", "8",
           "--aw-host", HOSTS, "--log_file_root", out]
    with open(f"{ROOT}/anchor/{label}.log", "w") as lg:
        subprocess.run(cmd, cwd=f"{D}/MobileWorld", env=env,
                       stdout=lg, stderr=subprocess.STDOUT)
    n = k = 0
    for d in glob.glob(os.path.join(out, "*/")):
        rp = os.path.join(d, "result.txt")
        if not os.path.exists(rp):
            continue
        try:
            sc = float(open(rp).read().split("score:")[1].split()[0])
        except Exception:  # noqa: BLE001
            continue
        n += 1
        k += 1 if sc > 0 else 0
    rec = {"round": rnd, "policy": policy, "judged": n, "success": k,
           "rate": k / max(n, 1), "t": time.time()}
    with open(CURVE, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"ANCHOR round={rnd} {policy}: {k}/{n} = {rec['rate']:.3f}", flush=True)


def main():
    os.makedirs(f"{ROOT}/anchor", exist_ok=True)
    if not any(json.loads(l).get("policy") == "recent" for l in open(CURVE)) \
            if os.path.exists(CURVE) else True:
        run_eval(-1, "recent", "recency_base")
    while True:
        r = cur_round()
        if r >= 0 and r % EVERY == 0 and r not in evaluated():
            run_eval(r, "learned", f"learned_r{r}")
        time.sleep(300)


if __name__ == "__main__":
    main()
