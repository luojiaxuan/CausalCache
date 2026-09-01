# note (luojiaxuan): Stage A 审计归约。核心问题(外审 20260901):
# teacher-forcing 的 Δlogp 是否预测贪心解码动作的正确性。四个上下文
# (空/recency/最优对/最差对)各有实测 u 与解码文本;解码与参考动作在
# 模型坐标系(0-1000)内做等价类匹配:动作类型(含 swipe/scroll/pull 与
# open/open_app 别名)+ 坐标容差。附带归约:逆序敏感度、协同项 γ 分布。
import glob, json, math, re, statistics as st

TOL_CLICK = 80
TOL_SWIPE = 150
ALIAS = {"scroll": "swipe", "pull": "swipe", "open_app": "open"}


def parse_action(text):
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text or "", re.S)
    if not m:
        return None
    try:
        args = json.loads(m.group(1)).get("arguments") or {}
    except Exception:  # noqa: BLE001
        return None
    a = args.get("action")
    if not a:
        return None
    return {"action": ALIAS.get(a, a),
            "coord": args.get("coordinate"), "coord2": args.get("coordinate2"),
            "text": (args.get("text") or "").strip().lower()}


def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1]) if p and q else 1e9


def match(dec, ref):
    if dec is None or ref is None:
        return None
    if dec["action"] != ref["action"]:
        return 0
    a = dec["action"]
    if a in ("click", "long_press"):
        return int(dist(dec["coord"], ref["coord"]) <= TOL_CLICK)
    if a == "swipe":
        if dec["coord"] and ref["coord"] and dec["coord2"] and ref["coord2"]:
            return int(dist(dec["coord"], ref["coord"]) <= TOL_SWIPE and
                       dist(dec["coord2"], ref["coord2"]) <= TOL_SWIPE)
        return int(bool(dec["coord"]) == bool(ref["coord"]))
    if a in ("type", "key", "open", "answer"):
        return int(dec["text"] == ref["text"] or
                   (len(ref["text"]) > 3 and ref["text"] in dec["text"]))
    return 1


def main():
    recs = []
    for f in sorted(glob.glob("/data01/jaxan/labels_audit_shard*.fixed.jsonl")) or \
         sorted(glob.glob("/data01/jaxan/labels_audit_shard*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("decode2") and r.get("cur_only") is not None:
                recs.append(r)
    print(f"审计状态数 {len(recs)}")

    ctxs = ["null", "recency", "best_pair", "worst_pair"]
    per_ctx = {c: [] for c in ctxs}
    pairs_cmp = []          # (du, dcorrect) 同状态跨上下文
    for r in recs:
        ref = parse_action(r["target"])
        if ref is None:
            continue
        us = {"null": r["cur_only"], "recency": r["recency"],
              "best_pair": r["pairs"].get(r["decode2"]["best_pair"]["key"]),
              "worst_pair": r["pairs"].get(r["decode2"]["worst_pair"]["key"])}
        cs = {}
        for c in ctxs:
            t = r["decode2"][c] if isinstance(r["decode2"][c], str) else r["decode2"][c]["text"]
            cs[c] = match(parse_action(t), ref)
        for c in ctxs:
            if cs[c] is not None and us[c] is not None:
                per_ctx[c].append((us[c], cs[c]))
        ks = [c for c in ctxs if cs[c] is not None and us[c] is not None]
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                pairs_cmp.append((us[ks[i]] - us[ks[j]], (cs[ks[i]] or 0) - (cs[ks[j]] or 0)))

    print("\n== 各上下文:平均 u 与解码正确率")
    for c in ctxs:
        xs = per_ctx[c]
        if xs:
            print(f"  {c:11s} u={st.mean(x for x, _ in xs):+.4f}  "
                  f"correct={st.mean(y for _, y in xs):.3f}  n={len(xs)}")

    concord = [(du, dc) for du, dc in pairs_cmp if abs(du) > 0.05 and dc != 0]
    if concord:
        agree = sum(1 for du, dc in concord if (du > 0) == (dc > 0))
        print(f"\n== 判决主指标:|Δu|>0.05 且正确性不同的上下文对 {len(concord)} 个,"
              f"方向一致 {agree} = {agree/len(concord):.1%}(50% = 无预测力)")
    xs = [du for du, _ in pairs_cmp]
    ys = [dc for _, dc in pairs_cmp]
    if len(xs) > 2 and st.pstdev(xs) > 0 and st.pstdev(ys) > 0:
        r_ = (st.mean(x * y for x, y in zip(xs, ys)) - st.mean(xs) * st.mean(ys)) / (
            st.pstdev(xs) * st.pstdev(ys))
        print(f"   Δu 与 Δcorrect 相关系数 r={r_:+.3f}  (对数 {len(xs)})")

    revd, flips = [], 0
    n_rev = 0
    for r in recs:
        pr = r.get("pairs_rev") or {}
        pv = {k: v for k, v in (r.get("pairs") or {}).items()
              if v is not None and pr.get(k) is not None}
        if len(pv) < 10:
            continue
        n_rev += 1
        revd.extend(abs(pv[k] - pr[k]) for k in pv)
        if max(pv, key=pv.get) != max({k: pr[k] for k in pv}, key=lambda k: pr[k]):
            flips += 1
    if revd:
        print(f"\n== 逆序敏感度:平均 |u_ij−u_ji| = {st.mean(revd):.4f}"
              f"(参照 oracle 增益 ~0.25);argmax 改变 {flips}/{n_rev}")

    gams = []
    for r in recs:
        u0 = r.get("cur_only")
        sg = r.get("singles") or {}
        for k, v in (r.get("pairs") or {}).items():
            a, b = k.split("_")
            if None in (v, sg.get(a), sg.get(b), u0):
                continue
            gams.append(v - sg[a] - sg[b] + u0)
    if gams:
        q = sorted(gams)
        print(f"\n== 协同项 γ=u_ij−u_i−u_j+u_∅:median {q[len(q)//2]:+.4f}  "
              f"P10 {q[len(q)//10]:+.4f}  P90 {q[-len(q)//10]:+.4f}  n={len(gams)}")
        print(f"   |γ|>0.1 的组合占 {st.mean(abs(g) > 0.1 for g in gams):.1%}"
              f"(小则单帧检索即可,不需要 pair 模型)")


if __name__ == "__main__":
    main()
