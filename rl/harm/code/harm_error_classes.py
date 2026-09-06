# note (luojiaxuan): history-harm Step 2 —— 把"不给图对、给图错"的伤害状态按错误形态分类(全部用现有标签 + 轨迹,零 GPU)。
# 类别(互斥,按先后判定):
#   finished_early  给图后输出 terminate/finished 而目标不是 → 早停
#   type_drift      动作类型变了(click→type 等)
#   stale_replay    动作与该轨迹**更早某步**的动作等价(复读过期动作;null 解码不与之等价)
#   past_grounding  同类型 click,坐标离目标 >TOL 且落在**更早某步的点击点** ±TOL 内(今昔混淆的代理)
#   coord_drift     同类型 click,坐标错但不靠近任何过去点击
#   other
import glob, json, math, sys
sys.path.insert(0, "/data01/jaxan")
from guiowl_oracle import parse_action, match, TOL_CLICK, iter_states
R = "/data01/jaxan/rl_v2"
states = {f"{s['dir']}|{s['step']}": s for s in iter_states([f"{R}/guiowl_base"], 12)}
hist_actions = {}
for k, s in states.items():
    import os
    traj = list(json.load(open(os.path.join(s["dir"], "traj.json"))).values())[0]["traj"]
    hist_actions[k] = [parse_action(t.get("prediction") or "") for t in traj if int(t["step"]) < s["step"]]

def classify(dec_txt, null_txt, ref, past):
    dec = parse_action(dec_txt); nul = parse_action(null_txt)
    if dec is None: return "unparseable"
    if dec["action"] in ("terminate", "finished", "answer") and ref["action"] not in ("terminate", "finished", "answer"): return "finished_early"
    if dec["action"] != ref["action"]: return "type_drift"
    for pa in past:
        if pa is not None and match(dec, pa) and not (nul is not None and match(nul, pa)): return "stale_replay"
    if dec["action"] in ("click", "long_press"):
        p = dec["coord"]; q = ref["coord"]
        if p and q and math.dist(p, q) > TOL_CLICK:
            for pa in past:
                if pa is not None and pa["action"] in ("click", "long_press") and pa["coord"] and math.dist(p, pa["coord"]) <= TOL_CLICK:
                    return "past_grounding"
            return "coord_drift"
    return "other"

def rows_from(paths):
    out = {}
    for f in sorted(glob.glob(paths)):
        for l in open(f):
            r = json.loads(l); k = f"{r['dir']}|{r['step']}"; out.setdefault(k, {}).update(r["decodes"])
    return out
dec = rows_from(f"{R}/guiowl_selfref.jsonl")
for f in glob.glob(f"{R}/judge_decode_base*.jsonl"):
    for l in open(f):
        r = json.loads(l); k = f"{r['dir']}|{r['step']}"; dec.setdefault(k, {}).update(r["decodes"])
PAIRS = [f"{a}_{b}" for a in range(6) for b in range(a + 1, 6)]
conds = {"recency": ["recency"], "uniform_pairs": PAIRS, "irrelevant": ["irrelevant"],
         "judge_glm": ["judge:judge_glm46v|direct|full"], "judge_qwen38": ["judge:qwen38_27b|direct|full"], "judge_30b": ["judge:judge|direct|full"]}
print(f"states={len(dec)}  伤害状态 = null 对 且 X 错;类别占比(%)")
hdr = ["finished_early", "type_drift", "stale_replay", "past_grounding", "coord_drift", "other", "unparseable"]
print(f"  {'条件':14s} {'伤害数':>6s} " + " ".join(f"{h:>14s}" for h in hdr))
for name, keys in conds.items():
    cnt = {h: 0 for h in hdr}; n = 0
    for k, d in dec.items():
        st = states.get(k)
        if st is None or "null" not in d: continue
        ref = parse_action(st["target"])
        if ref is None: continue
        if not match(parse_action(d["null"]), ref): continue
        for key in keys:
            if key not in d: continue
            if match(parse_action(d[key]), ref): continue
            n += 1; cnt[classify(d[key], d["null"], ref, hist_actions[k])] += 1
    if n: print(f"  {name:14s} {n:6d} " + " ".join(f"{100*cnt[h]/n:14.1f}" for h in hdr))
