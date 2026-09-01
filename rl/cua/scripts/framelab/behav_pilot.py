# note (luojiaxuan): 行为天花板试点。teacher-forcing 代理双双出局后,直接
# 用行为量帧选择的价值:每状态在 23 个上下文(空+6单+15对)下各做一次
# 带 system 的贪心解码,与参考动作做等价类匹配。产出:行为口径的
# oracle-vs-recency 差、"recency 错而某对能救"的比例、行为标签的动态范围。
import glob, itertools, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/data01/jaxan")
from frame_label_mass import _post, _content  # noqa: E402

BASE = sys.argv[1]
SHARD = int(sys.argv[2])
LIMIT = 150
SYSTEM = json.loads(open("/data01/jaxan/sglang-omni-rl/sft_random_s_v1.jsonl")
                    .readline())["system"]


def greedy_sys(imgs, goal):
    out = _post(BASE, "/chat/completions", {
        "model": "probe", "max_tokens": 96, "temperature": 0.0,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": _content(imgs, goal)}]})
    return out["choices"][0]["message"]["content"]


def contexts(rec):
    d = os.path.join(rec["dir"], "screenshots")
    shots = {int(re.search(r"-(\d+)\.png$", f).group(1)): f
             for f in glob.glob(os.path.join(d, "*.png"))}
    k = rec["step"]
    cands = [shots[n] for n in rec["cand_files"]]
    cur = shots[k]
    yield "null", [cur]
    yield "recency", [shots[k - 1], shots[k - 2], cur]
    for i in range(6):
        yield f"s{i}", [cands[i], cur]
    for a, b in itertools.combinations(range(6), 2):
        yield f"{a}_{b}", [cands[a], cands[b], cur]


def main():
    src = f"/data01/jaxan/labels_audit_shard{SHARD}.fixed.jsonl"
    dst = f"/data01/jaxan/behav_pilot_shard{SHARD}.jsonl"
    done = set()
    if os.path.exists(dst):
        for line in open(dst):
            try:
                done.add(json.loads(line)["dir"] + "|" + str(json.loads(line)["step"]))
            except Exception:  # noqa: BLE001
                continue
    recs = [json.loads(l) for l in open(src)][:LIMIT]
    recs = [r for r in recs if f"{r['dir']}|{r['step']}" not in done]
    print(f"shard {SHARD}: {len(recs)}", flush=True)
    lock = threading.Lock()
    n = 0

    def work(rec):
        nonlocal n
        try:
            out = {"dir": rec["dir"], "step": rec["step"], "goal": rec["goal"],
                   "target": rec["target"], "decodes": {}}
            for name, imgs in contexts(rec):
                out["decodes"][name] = greedy_sys(imgs, rec["goal"])
        except Exception as e:  # noqa: BLE001
            print(f"ERR {rec['dir']}|{rec['step']}: {str(e)[:100]}", flush=True)
            return
        with lock:
            with open(dst, "a") as f:
                f.write(json.dumps(out) + "\n")
            n += 1
            if n % 20 == 0:
                print(f"  {n}/{len(recs)}", flush=True)

    with ThreadPoolExecutor(max_workers=10) as ex:
        list(ex.map(work, recs))
    print(f"BEHAV_DONE {SHARD}", flush=True)


if __name__ == "__main__":
    main()
