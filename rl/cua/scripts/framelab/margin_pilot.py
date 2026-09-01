# note (luojiaxuan): margin 标签试点(外审 20260901 预案:raw likelihood 闸门
# 未过时的后备)。对每个审计状态的 4 个已解码上下文,给若干 hard negative
# 动作打 teacher-forcing 分,margin = u_ref − max_neg u_neg;闸门同口径重跑
# 由离线脚本做。负样本 = 本状态各上下文解码错误的动作(自然负样本)+
# 参考动作坐标平移 ±150 的合成负样本,至多 3 个。
import glob, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/data01/jaxan")
from frame_label_mass import score2  # noqa: E402
from audit_analysis import parse_action, match  # noqa: E402

BASE = sys.argv[1]
SHARD = int(sys.argv[2])


def negatives(rec):
    ref = parse_action(rec["target"])
    if ref is None:
        return []
    negs = []
    for c in ("null", "recency", "best_pair", "worst_pair"):
        t = rec["decode2"][c] if isinstance(rec["decode2"][c], str) else rec["decode2"][c]["text"]
        p = parse_action(t)
        if p is not None and match(p, ref) == 0 and t not in negs:
            negs.append(t)
    if ref.get("coord"):
        x, y = ref["coord"]
        shifted = re.sub(r'"coordinate":\s*\[[0-9.,\s]+\]',
                         f'"coordinate": [{min(999, x + 150)}, {min(999, y + 150)}]',
                         rec["target"], count=1)
        if shifted != rec["target"]:
            negs.append(shifted)
    return negs[:3]


def ctx_imgs(rec, c):
    d = os.path.join(rec["dir"], "screenshots")
    shots = {int(re.search(r"-(\d+)\.png$", f).group(1)): f
             for f in glob.glob(os.path.join(d, "*.png"))}
    k = rec["step"]
    cands = [shots[n] for n in rec["cand_files"]]
    cur = shots[k]
    if c == "null":
        return [cur]
    if c == "recency":
        return [shots[k - 1], shots[k - 2], cur]
    key = rec["decode2"][c]["key"]
    a, b = (int(x) for x in key.split("_"))
    return [cands[a], cands[b], cur]


def main():
    src = f"/data01/jaxan/labels_audit_shard{SHARD}.fixed.jsonl"
    dst = f"/data01/jaxan/margin_pilot_shard{SHARD}.jsonl"
    done = set()
    if os.path.exists(dst):
        for line in open(dst):
            try:
                r = json.loads(line)
                done.add(f"{r['dir']}|{r['step']}")
            except Exception:  # noqa: BLE001
                continue
    recs = []
    for line in open(src):
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if r.get("decode2") and f"{r['dir']}|{r['step']}" not in done:
            recs.append(r)
    print(f"shard {SHARD}: 试点 {len(recs)}(已完成 {len(done)})", flush=True)
    lock = threading.Lock()
    n = 0

    def work(rec):
        nonlocal n
        try:
            negs = negatives(rec)
            out = {"dir": rec["dir"], "step": rec["step"], "n_negs": len(negs),
                   "neg_scores": {}, "neg_toks": {}}
            if negs:
                for c in ("null", "recency", "best_pair", "worst_pair"):
                    imgs = ctx_imgs(rec, c)
                    ss, ts = [], []
                    for ng in negs:
                        m, tk = score2(BASE, "probe", imgs, rec["goal"], ng)
                        ss.append(m)
                        ts.append(tk)
                    out["neg_scores"][c] = ss
                    out["neg_toks"][c] = ts
        except Exception as e:  # noqa: BLE001
            print(f"ERR {rec['dir']}|{rec['step']}: {str(e)[:100]}", flush=True)
            return
        with lock:
            with open(dst, "a") as f:
                f.write(json.dumps(out) + "\n")
            n += 1
            if n % 25 == 0:
                print(f"  {n}/{len(recs)}", flush=True)

    with ThreadPoolExecutor(max_workers=10) as ex:
        list(ex.map(work, recs))
    print(f"PILOT_DONE {SHARD}", flush=True)


if __name__ == "__main__":
    main()
