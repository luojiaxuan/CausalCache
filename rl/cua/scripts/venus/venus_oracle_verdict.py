# note (luojiaxuan): G2 归约。Venus 动作串 → (类型, 坐标, 文本) 等价类匹配:点击类坐标在 0–999 网格上距离 ≤ TOL;
# 滑动类起止各 ≤ TOL;输入/启动/回答类小写相等或参考文本长度>3 时为解码文本子串;按键/等待/结束类型相同即可。
# 输出:recency / 空 / 随机对均值 / ∃正确对 / ∃正确单帧 / recency 错时可救回率,并按 heldout-39 与 train 分报。
import ast, glob, json, math, re, statistics as st, sys
TOL = 60
split = json.load(open("/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json"))
HELD = set(split["heldout"])


def parse(a):
    m = re.match(r"(\w+)\((.*)\)\s*$", (a or "").strip(), re.S)
    if not m:
        return None
    name, params = m.group(1), {}
    if m.group(2).strip():
        try:
            tree = ast.parse("_(" + m.group(2).replace("\n", "\\n") + ")", mode="eval")
            params = {kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords}
        except (SyntaxError, ValueError):
            params = {}
    return name, params


def match(dec, ref):
    if dec is None or ref is None:
        return None
    (dn, dp), (rn, rp) = dec, ref
    if dn != rn:
        return 0
    if dn in ("Click", "DoubleClick", "LongPress"):
        a, b = dp.get("point") or dp.get("box"), rp.get("point") or rp.get("box")
        return int(bool(a and b) and math.dist(a, b) <= TOL)
    if dn in ("Swipe", "Drag"):
        ok = all(k in dp and k in rp for k in ("start", "end"))
        return int(ok and math.dist(dp["start"], rp["start"]) <= TOL and math.dist(dp["end"], rp["end"]) <= TOL)
    if dn == "Finished":
        # note (luojiaxuan): 结束动作的 content 是自由文本总结,类型相同即等价。
        return 1
    if dn in ("Type", "LaunchApp", "Answer", "CallUser"):
        key = "app" if dn == "LaunchApp" else "content"
        d, r = str(dp.get(key, "")).strip().lower(), str(rp.get(key, "")).strip().lower()
        return int(d == r or (len(r) > 3 and r in d))
    return 1


def summarize(recs, label):
    rec = null = []; rec, null, mean_pair, best_pair, best_single = [], [], [], [], []
    rescue = rec_wrong = 0
    for r in recs:
        ref = parse(r["target"])
        cm = {k: match(parse(v), ref) for k, v in r["decodes"].items()}
        if cm.get("recency") is None or ref is None:
            continue
        pairs = [cm[k] or 0 for k in cm if "_" in k and not k.startswith("s")]
        singles = [cm[f"s{i}"] or 0 for i in range(6) if cm.get(f"s{i}") is not None]
        rec.append(cm["recency"] or 0); null.append(cm.get("null") or 0)
        mean_pair.append(st.mean(pairs)); best_pair.append(max(pairs)); best_single.append(max(singles))
        if not cm["recency"]:
            rec_wrong += 1; rescue += max(pairs)
    n = len(rec)
    if not n:
        print(f"[{label}] no states"); return
    print(f"[{label}] states={n} recency={st.mean(rec):.3f} null={st.mean(null):.3f} random_pair={st.mean(mean_pair):.3f} "
          f"oracle_pair={st.mean(best_pair):.3f} oracle_single={st.mean(best_single):.3f} "
          f"rescue={rescue}/{rec_wrong}={100*rescue/max(rec_wrong,1):.0f}%")


recs = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
summarize(recs, "all")
summarize([r for r in recs if r["task"] in HELD], "heldout39")
summarize([r for r in recs if r["task"] not in HELD], "train78")
