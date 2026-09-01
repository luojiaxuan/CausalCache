# note (luojiaxuan): 行为 argmax 闸门·正式版(外审 20260901:Stage B 的
# go/no-go)。每状态:(1) 同状态共享负样本池(与试点同一构造,确定性
# 可复现)在全部 15 个双帧上下文下打分,margin = u_ref − τ·lse(负样本);
# (2) 解码 margin-argmax 对,与 recency 解码比正确率;(3) NL 规范化鲁棒性:
# 把参考与负样本的 NL 描述替换为由 JSON 生成的模板句,4 个已解码上下文下
# 重算 margin,检验排序是否为语言形式伪影。
import glob, json, math, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/data01/jaxan")
from frame_label_mass import score2, greedy as _greedy_nosys, _post, _content  # noqa: E402
from margin_pilot import negatives, ctx_imgs  # noqa: E402
from audit_analysis import parse_action  # noqa: E402

BASE = sys.argv[1]
SHARD = int(sys.argv[2])
TAU = 0.25
SYSTEM = json.loads(open("/data01/jaxan/sglang-omni-rl/sft_random_s_v1.jsonl")
                    .readline())["system"]


def greedy_sys(imgs, goal):
    out = _post(BASE, "/chat/completions", {
        "model": "probe", "max_tokens": 96, "temperature": 0.0,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": _content(imgs, goal)}]})
    return out["choices"][0]["message"]["content"]


def canon(text):
    """NL 描述替换为 JSON 生成的模板句,动作 JSON 原样保留。"""
    m = re.search(r"(<tool_call>.*?</tool_call>)", text or "", re.S)
    if not m:
        return None
    p = parse_action(text)
    if p is None:
        return None
    nl = f"Action: {p['action']}"
    if p.get("coord"):
        nl += f" at ({p['coord'][0]}, {p['coord'][1]})"
    if p.get("text"):
        nl += f" with '{p['text']}'"
    return nl + ".\n" + m.group(1)


def lse_margin(ref, negs):
    negs = [x for x in negs if x is not None]
    if ref is None or not negs:
        return None
    return ref - TAU * math.log(sum(math.exp(x / TAU) for x in negs))


def pair_imgs(rec, key):
    d = os.path.join(rec["dir"], "screenshots")
    shots = {int(re.search(r"-(\d+)\.png$", f).group(1)): f
             for f in glob.glob(os.path.join(d, "*.png"))}
    cands = [shots[n] for n in rec["cand_files"]]
    a, b = (int(x) for x in key.split("_"))
    return [cands[a], cands[b], shots[rec["step"]]]


def work_state(rec):
    negs = negatives(rec)
    if not negs:
        return None
    out = {"dir": rec["dir"], "step": rec["step"], "n_negs": len(negs),
           "pair_margins": {}, "canon": {}}
    for key in rec["pairs"]:
        if rec["pairs"][key] is None:
            continue
        imgs = pair_imgs(rec, key)
        ns = [score2(BASE, "probe", imgs, rec["goal"], ng)[0] for ng in negs]
        out["pair_margins"][key] = lse_margin(rec["pairs"][key], ns)
    valid = {k: v for k, v in out["pair_margins"].items() if v is not None}
    if not valid:
        return None
    out["argmax_key"] = max(valid, key=valid.get)
    out["argmax_decode"] = greedy_sys(pair_imgs(rec, out["argmax_key"]), rec["goal"])
    cref = canon(rec["target"])
    cnegs = [c for c in (canon(n) for n in negs) if c]
    if cref and cnegs:
        for c in ("null", "recency", "best_pair", "worst_pair"):
            imgs = ctx_imgs(rec, c)
            r0 = score2(BASE, "probe", imgs, rec["goal"], cref)[0]
            ns = [score2(BASE, "probe", imgs, rec["goal"], ng)[0] for ng in cnegs]
            out["canon"][c] = lse_margin(r0, ns)
    return out


def main():
    src = f"/data01/jaxan/labels_audit_shard{SHARD}.fixed.jsonl"
    dst = f"/data01/jaxan/argmax_gate_shard{SHARD}.jsonl"
    done = set()
    if os.path.exists(dst):
        for line in open(dst):
            try:
                r = json.loads(line)
                done.add(f"{r['dir']}|{r['step']}")
            except Exception:  # noqa: BLE001
                continue
    recs = [json.loads(l) for l in open(src)]
    recs = [r for r in recs if r.get("decode2") and f"{r['dir']}|{r['step']}" not in done]
    print(f"shard {SHARD}: {len(recs)}(已完成 {len(done)})", flush=True)
    lock = threading.Lock()
    n = 0

    def work(rec):
        nonlocal n
        try:
            out = work_state(rec)
        except Exception as e:  # noqa: BLE001
            print(f"ERR {rec['dir']}|{rec['step']}: {str(e)[:100]}", flush=True)
            return
        if out is None:
            return
        with lock:
            with open(dst, "a") as f:
                f.write(json.dumps(out) + "\n")
            n += 1
            if n % 20 == 0:
                print(f"  {n}/{len(recs)}", flush=True)

    with ThreadPoolExecutor(max_workers=10) as ex:
        list(ex.map(work, recs))
    print(f"GATE_DONE {SHARD}", flush=True)


if __name__ == "__main__":
    main()
