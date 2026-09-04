# note (luojiaxuan): G2 的赢家诅咒对照与救回归因。∃-over-15-pairs 在基础复现率约 90% 时天然接近 100%,
# 所以再看:(1) 每状态 23 个上下文里出现几种不同动作(等价类去重)——多样性低说明上下文几乎不改变输出;
# (2) recency 错的状态里,空上下文(不给历史图)是否也对——若是,则救回来自"扰动"而非"记忆";
# (3) 救回对是否必须含远帧(帧龄 ≥5)——"远程记忆"的证据只能来自这一类。
import json, statistics as st, sys
sys.path.insert(0, "/data01/jaxan")
from venus_oracle_verdict import parse, match, HELD  # noqa: E402

recs = [json.loads(l) for f in sys.argv[1:] for l in open(f)]

def pt(v):
    # note (luojiaxuan): 模型偶尔把坐标写成标量或字符串(系统边界:解析模型输出),归到 (0,0) 桶。
    return v if isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v) else (0, 0)

def canon(a):
    p = parse(a)
    if p is None: return None
    n, q = p
    if n in ("Click", "DoubleClick", "LongPress"):
        p0 = pt(q.get("point") or q.get("box")); return (n, round(p0[0] / 60), round(p0[1] / 60))
    if n in ("Swipe", "Drag"):
        s, e = pt(q.get("start")), pt(q.get("end")); return (n, round(s[0]/60), round(s[1]/60), round(e[0]/60), round(e[1]/60))
    if n in ("Type", "LaunchApp", "Answer", "CallUser"):
        return (n, str(q.get("content", q.get("app", ""))).strip().lower()[:40])
    return (n,)

def run(label, rs):
    div, null_rescue, far_only, near_ok, rec_wrong = [], 0, 0, 0, 0
    for r in rs:
        ref = parse(r["target"])
        cm = {k: match(parse(v), ref) for k, v in r["decodes"].items()}
        if ref is None or cm.get("recency") is None: continue
        div.append(len({canon(v) for v in r["decodes"].values()} - {None}))
        if cm["recency"]: continue
        rec_wrong += 1
        k, c = r["step"], r["cand_idx"]
        wins = [key for key in cm if "_" in key and not key.startswith("s") and cm[key]]
        if cm.get("null"): null_rescue += 1
        def older_age(key):
            a, b = (int(x) for x in key.split("_")); return (k - 1) - min(c[a], c[b])
        def has_near(key):
            a, b = (int(x) for x in key.split("_")); return max(c[a], c[b]) >= k - 3
        if wins:
            if all(older_age(w) >= 5 for w in wins): far_only += 1
            if any(has_near(w) for w in wins): near_ok += 1
    n = len(div)
    print(f"[{label}] states={n} distinct_actions/state: median={st.median(div)} mean={st.mean(div):.2f} share_single={sum(d==1 for d in div)/n:.2f} | "
          f"recency_wrong={rec_wrong} null_also_right={null_rescue} rescued_only_by_far_pairs(age>=5)={far_only} rescued_by_pair_with_near_frame={near_ok}")

run("all", recs); run("heldout39", [r for r in recs if r["task"] in HELD]); run("train78", [r for r in recs if r["task"] not in HELD])
