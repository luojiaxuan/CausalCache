# note (luojiaxuan): 贪心侦察评测。RL 循环已暂停,占用三个训练槽(24 台模拟器)。
# 阶段 1(服务当前载 RL 权重 pv22):heldout-20×learned-greedy、train-20×learned-greedy、
# train-20×recency 并行;阶段 2 把服务切回预训练初始权重(energy_final.pt,pv 9001):
# heldout-20×init-greedy、train-20×init-greedy 并行。结果写 greedy/results.jsonl。
import glob, json, os, random, shutil, subprocess, sys, time, urllib.request

D = "/data01/jaxan/mw"
ROOT = "/data01/jaxan/rl_v2"
OUT = f"{ROOT}/greedy"
LLM = os.environ["RL_LLM_URL"]
SEL = os.environ["RL_SEL_URL"]
split = json.load(open("/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json"))
TRAIN = sorted(split.get("train") or split.get("train_tasks"))
HELDOUT = sorted(random.Random(20260903).sample(sorted(split["heldout"]), 20))
TRAIN20 = sorted(random.Random(20260904).sample(TRAIN, 20))
ALL_PORTS = [l.split("\t")[3].strip() for l in
             open("/data01/jaxan/sglang-omni-rl/pool_ports.tsv") if l.strip()]


def launch(label, policy, greedy, tasks, slot):
    out = f"{OUT}/{label}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ)
    env.update({"CC_HISTORY_N": "3", "CC_FRAME_POLICY": policy, "CC_SELECTOR_URL": SEL,
                "CC_EP_TAG": label, "CC_SEL_GREEDY": "1" if greedy else "0",
                "PYTHONPATH": "/data01/jaxan/pyshim"})
    ports = ALL_PORTS[slot * 8:(slot + 1) * 8]
    hosts = ",".join(f"http://127.0.0.1:{p}" for p in ports)
    cmd = ["timeout", "5400", "uv", "run", "mw", "eval", "--agent_type", "gui_owl_1_5",
           "--task", ",".join(tasks), "--max_round", "50", "--model_name", "gui-owl",
           "--llm_base_url", LLM, "--api_key", "EMPTY", "--step_wait_time", "3",
           "--max-concurrency", "8", "--aw-host", hosts, "--log_file_root", out]
    lg = open(f"{OUT}/{label}.log", "w")
    print(f"[launch] {label} slot={slot} n_tasks={len(tasks)}", flush=True)
    return subprocess.Popen(cmd, cwd=f"{D}/MobileWorld", env=env, stdout=lg, stderr=subprocess.STDOUT)


def harvest(label, policy, greedy, weights):
    out = f"{OUT}/{label}"
    n = k = 0
    per = {}
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
        per[os.path.basename(d.rstrip("/"))] = int(sc > 0)
        shutil.rmtree(os.path.join(d, "screenshots"), ignore_errors=True)
    rec = {"label": label, "policy": policy, "greedy": greedy, "weights": weights,
           "judged": n, "success": k, "rate": k / max(n, 1), "per_task": per, "t": time.time()}
    with open(f"{OUT}/results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"RESULT {label}: {k}/{n} = {rec['rate']:.3f}", flush=True)


def reload_weights(path, pv):
    req = urllib.request.Request(SEL.rstrip("/") + "/reload",
                                 data=json.dumps({"path": path, "pv": pv}).encode(),
                                 headers={"Content-Type": "application/json"})
    print("reload:", urllib.request.urlopen(req, timeout=120).read().decode(), flush=True)


def phase(specs, weights):
    procs = [(s, launch(s[0], s[1], s[2], s[3], i)) for i, s in enumerate(specs)]
    for s, p in procs:
        p.wait()
        print(f"[exit] {s[0]} rc={p.returncode}", flush=True)
        harvest(s[0], s[1], s[2], weights)


os.makedirs(OUT, exist_ok=True)
phase([("heldout20_learned_greedy_pv22", "learned", True, HELDOUT),
       ("train20_learned_greedy_pv22", "learned", True, TRAIN20),
       ("train20_recency", "recent", False, TRAIN20)], "rl_pv22")
reload_weights("/data01/jaxan/selector_feats/energy_final.pt", 9001)
phase([("heldout20_init_greedy", "learned", True, HELDOUT),
       ("train20_init_greedy", "learned", True, TRAIN20)], "pretrain_init")
print("GREEDY_RECON_DONE", flush=True)
