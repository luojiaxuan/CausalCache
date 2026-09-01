# note (luojiaxuan): margin 口径变体扫描 + 分对型拆解,全离线。
# M1 = 全 token margin(基线);M2 = 动作段 margin(<tool_call> 起)。
import glob, json, sys
from transformers import AutoTokenizer
sys.path.insert(0, "/data01/jaxan")
from audit_analysis import parse_action, match  # noqa: E402

tok = AutoTokenizer.from_pretrained(
    "/data04/jaxan/models/GUI-Owl-1.5-8B-Instruct", trust_remote_code=True)
TC = tok.convert_tokens_to_ids("<tool_call>")

def u_span(toks, action_only):
    if not toks:
        return None
    ids = [t for t, _ in toks]
    if action_only:
        if TC not in ids:
            return None
        toks = toks[ids.index(TC):]
    return sum(lp for _, lp in toks) / len(toks)

aud, mg = {}, {}
for f in sorted(glob.glob("/data01/jaxan/labels_audit_shard*.fixed.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if r.get("decode2"):
            aud[f"{r['dir']}|{r['step']}"] = r
for f in sorted(glob.glob("/data01/jaxan/margin_pilot_shard*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if r.get("neg_scores"):
            mg[f"{r['dir']}|{r['step']}"] = r

CTX = ("null", "recency", "best_pair", "worst_pair")

def ref_toks(r, c):
    if c == "null":
        return r["cur_only_tok"]
    if c == "recency":
        return r["recency_tok"]
    return r["pairs_tok"].get(r["decode2"][c]["key"])

for action_only, tag in ((False, "M1 全 token margin"), (True, "M2 动作段 margin")):
    cmp_pairs = []
    for k, r in aud.items():
        m = mg.get(k)
        if not m or not m.get("n_negs"):
            continue
        ref = parse_action(r["target"])
        if ref is None:
            continue
        mar, cs = {}, {}
        for c in CTX:
            rt = ref_toks(r, c)
            ur = u_span(rt, action_only)
            negs = [u_span(t, action_only) for t in m["neg_toks"].get(c, [])]
            negs = [x for x in negs if x is not None]
            if ur is None or not negs:
                continue
            mar[c] = ur - max(negs)
            t = r["decode2"][c] if isinstance(r["decode2"][c], str) else r["decode2"][c]["text"]
            cs[c] = match(parse_action(t), ref)
        ks = [c for c in mar if cs.get(c) is not None]
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                dm = mar[ks[i]] - mar[ks[j]]
                dc = (cs[ks[i]] or 0) - (cs[ks[j]] or 0)
                if abs(dm) > 0.05 and dc != 0:
                    cmp_pairs.append((frozenset((ks[i], ks[j])), dm, dc))
    agree = sum(1 for _, dm, dc in cmp_pairs if (dm > 0) == (dc > 0))
    print(f"== {tag}: 总一致率 {agree}/{len(cmp_pairs)} = {100*agree/max(len(cmp_pairs),1):.1f}%")
    import collections
    by = collections.defaultdict(lambda: [0, 0])
    for pt, dm, dc in cmp_pairs:
        key = "+".join(sorted(x.replace("_pair", "") for x in pt))
        by[key][1] += 1
        if (dm > 0) == (dc > 0):
            by[key][0] += 1
    for key in sorted(by):
        a, n = by[key]
        print(f"     {key:22s} {a}/{n} = {100*a/n:.0f}%")
