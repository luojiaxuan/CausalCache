# note (luojiaxuan): 行为 argmax 闸门判决 + NL 规范化鲁棒性。
import glob, json, math, statistics as st, sys
sys.path.insert(0, "/data01/jaxan")
from audit_analysis import parse_action, match  # noqa: E402

aud, gate = {}, {}
for f in sorted(glob.glob("/data01/jaxan/labels_audit_shard*.fixed.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if r.get("decode2"):
            aud[f"{r['dir']}|{r['step']}"] = r
for f in sorted(glob.glob("/data01/jaxan/argmax_gate_shard*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        gate[f"{r['dir']}|{r['step']}"] = r

sel_c, rec_c = [], []
overlap_u_best = 0
for k, g in gate.items():
    r = aud.get(k)
    if not r or not g.get("argmax_decode"):
        continue
    ref = parse_action(r["target"])
    if ref is None:
        continue
    cm = match(parse_action(g["argmax_decode"]), ref)
    t = r["decode2"]["recency"]
    cr = match(parse_action(t), ref)
    if cm is None or cr is None:
        continue
    sel_c.append(cm)
    rec_c.append(cr)
    if g["argmax_key"] == r["decode2"]["best_pair"]["key"]:
        overlap_u_best += 1
n = len(sel_c)
d = st.mean(sel_c) - st.mean(rec_c)
se = math.sqrt(st.pstdev([a - b for a, b in zip(sel_c, rec_c)]) ** 2 / n)
print(f"== 行为 argmax 闸门(n={n},margin-argmax 与 u-best 重合 {overlap_u_best})")
print(f"   margin-argmax 对解码正确率: {st.mean(sel_c):.3f}")
print(f"   recency 解码正确率:        {st.mean(rec_c):.3f}")
print(f"   配对差: {d:+.3f}  se≈{se:.3f}  z≈{d/max(se,1e-9):+.1f}")

# NL 规范化:canon margin 与原 margin 的排序保持度 + canon margin 的行为预测力
mg = {}
for f in sorted(glob.glob("/data01/jaxan/margin_pilot_shard*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if r.get("neg_scores"):
            mg[f"{r['dir']}|{r['step']}"] = r
TAU = 0.25
CTX = ("null", "recency", "best_pair", "worst_pair")
keep, tot = 0, 0
cmp_pairs = []
for k, g in gate.items():
    r, m = aud.get(k), mg.get(k)
    if not r or not m or len(g.get("canon") or {}) < 3:
        continue
    ref = parse_action(r["target"])
    if ref is None:
        continue
    us = {"null": r["cur_only"], "recency": r["recency"],
          "best_pair": r["pairs"].get(r["decode2"]["best_pair"]["key"]),
          "worst_pair": r["pairs"].get(r["decode2"]["worst_pair"]["key"])}
    orig = {}
    for c in CTX:
        negs = [x for x in m["neg_scores"].get(c, []) if x is not None]
        if us[c] is None or not negs:
            continue
        orig[c] = us[c] - TAU * math.log(sum(math.exp(x / TAU) for x in negs))
    cs = {}
    for c in CTX:
        t = r["decode2"][c] if isinstance(r["decode2"][c], str) else r["decode2"][c]["text"]
        cs[c] = match(parse_action(t), ref)
    can = g["canon"]
    ks = [c for c in CTX if c in orig and can.get(c) is not None]
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = ks[i], ks[j]
            tot += 1
            if (orig[a] > orig[b]) == (can[a] > can[b]):
                keep += 1
            if cs.get(a) is not None and cs.get(b) is not None:
                dm = can[a] - can[b]
                dc = (cs[a] or 0) - (cs[b] or 0)
                if abs(dm) > 0.05 and dc != 0:
                    cmp_pairs.append((dm, dc))
agree = sum(1 for dm, dc in cmp_pairs if (dm > 0) == (dc > 0))
print(f"\n== NL 规范化鲁棒性")
print(f"   canon 与原 margin 排序保持: {keep}/{tot} = {100*keep/max(tot,1):.1f}%")
print(f"   canon margin 的方向一致率:  {agree}/{len(cmp_pairs)} = {100*agree/max(len(cmp_pairs),1):.1f}%"
      f"  (原 margin 64.8%)")
