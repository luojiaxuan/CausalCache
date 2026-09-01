# note (luojiaxuan): 闸门救援分析。把效用从"target 全 token 均值"改为
# "<tool_call> 起的动作 token 均值"重算,检验帧增益是否集中在动作段、
# 动作段 u 是否更能预测解码正确性。纯 CPU,复用已存逐 token 日志。
import glob, json, statistics as st, sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "/data04/jaxan/models/GUI-Owl-1.5-8B-Instruct", trust_remote_code=True)
TC = tok.convert_tokens_to_ids("<tool_call>")
print("tool_call token id:", TC)

def u_span(toks, action_only):
    if not toks:
        return None
    ids = [t for t, _ in toks]
    if action_only:
        if TC not in ids:
            return None
        toks = toks[ids.index(TC):]
    return sum(lp for _, lp in toks) / len(toks)

sys.path.insert(0, "/data01/jaxan")
from audit_analysis import parse_action, match  # noqa: E402

recs = []
for f in sorted(glob.glob("/data01/jaxan/labels_audit_shard*.fixed.jsonl")):
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if r.get("decode2") and r.get("cur_only_tok"):
            recs.append(r)
print(f"状态数 {len(recs)}")

for action_only, tag in ((False, "全 token(原口径,对照)"), (True, "仅动作段(<tool_call> 起)")):
    per = {}
    cmp_pairs = []
    gains, base_vals = [], []
    for r in recs:
        ref = parse_action(r["target"])
        if ref is None:
            continue
        us = {"null": u_span(r["cur_only_tok"], action_only),
              "recency": u_span(r["recency_tok"], action_only),
              "best_pair": u_span(r["pairs_tok"].get(r["decode2"]["best_pair"]["key"]), action_only),
              "worst_pair": u_span(r["pairs_tok"].get(r["decode2"]["worst_pair"]["key"]), action_only)}
        cs = {}
        for c in us:
            t = r["decode2"][c] if isinstance(r["decode2"][c], str) else r["decode2"][c]["text"]
            cs[c] = match(parse_action(t), ref)
        for c in us:
            if us[c] is not None and cs[c] is not None:
                per.setdefault(c, []).append((us[c], cs[c]))
        ks = [c for c in us if us[c] is not None and cs[c] is not None]
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                du = us[ks[i]] - us[ks[j]]
                dc = (cs[ks[i]] or 0) - (cs[ks[j]] or 0)
                if abs(du) > 0.05 and dc != 0:
                    cmp_pairs.append((du, dc))
        # 动作段口径下的 oracle-recency 帧增益
        pav = {k: u_span(v, action_only) for k, v in r["pairs_tok"].items()}
        pav = {k: v for k, v in pav.items() if v is not None}
        rv = u_span(r["recency_tok"], action_only)
        if len(pav) == 15 and rv is not None:
            gains.append(max(pav.values()) - rv)
            base_vals.append(rv)
    agree = sum(1 for du, dc in cmp_pairs if (du > 0) == (dc > 0))
    print(f"\n== {tag}")
    for c in ("null", "recency", "best_pair", "worst_pair"):
        xs = per.get(c, [])
        if xs:
            print(f"   {c:11s} u={st.mean(x for x,_ in xs):+.4f} correct={st.mean(y for _,y in xs):.3f}")
    print(f"   闸门:方向一致 {agree}/{len(cmp_pairs)} = {100*agree/max(len(cmp_pairs),1):.1f}%")
    if gains:
        print(f"   oracle−recency 帧增益(本口径):{st.mean(gains):+.4f}  (recency 绝对值 {st.mean(base_vals):.3f})")
