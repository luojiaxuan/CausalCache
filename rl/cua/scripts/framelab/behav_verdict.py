# note (luojiaxuan): 行为天花板归约。全部量为确定性贪心解码的等价类匹配。
import glob, json, statistics as st, sys
sys.path.insert(0, "/data01/jaxan")
from audit_analysis import parse_action, match  # noqa: E402

recs = []
for f in sorted(glob.glob("/data01/jaxan/behav_pilot_shard*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if len(r.get("decodes") or {}) >= 20:
            recs.append(r)
print(f"状态数 {len(recs)}")
pair_keys = [k for k in recs[0]["decodes"] if "_" in k and not k.startswith("s")]
res = {"rec": [], "null": [], "best_pair_exists": [], "rescue": 0, "n_rec_wrong": 0,
       "mean_pair": [], "best_single": []}
for r in recs:
    ref = parse_action(r["target"])
    if ref is None:
        continue
    cm = {k: match(parse_action(v), ref) for k, v in r["decodes"].items()}
    if cm.get("recency") is None:
        continue
    rec_ok = cm["recency"] or 0
    pairs = [cm[k] or 0 for k in pair_keys if cm.get(k) is not None]
    singles = [cm[f"s{i}"] or 0 for i in range(6) if cm.get(f"s{i}") is not None]
    if not pairs:
        continue
    res["rec"].append(rec_ok)
    res["null"].append(cm.get("null") or 0)
    res["best_pair_exists"].append(1 if max(pairs) else 0)
    res["mean_pair"].append(st.mean(pairs))
    res["best_single"].append(max(singles) if singles else 0)
    if not rec_ok:
        res["n_rec_wrong"] += 1
        if max(pairs):
            res["rescue"] += 1
n = len(res["rec"])
print(f"recency 正确率           : {st.mean(res['rec']):.3f}")
print(f"空上下文正确率           : {st.mean(res['null']):.3f}")
print(f"随机对正确率(均值)     : {st.mean(res['mean_pair']):.3f}")
print(f"∃正确对(行为 oracle)   : {st.mean(res['best_pair_exists']):.3f}")
print(f"∃正确单帧               : {st.mean(res['best_single']):.3f}")
print(f"recency 错时可被某对救回 : {res['rescue']}/{res['n_rec_wrong']}"
      f" = {100*res['rescue']/max(res['n_rec_wrong'],1):.0f}%")
