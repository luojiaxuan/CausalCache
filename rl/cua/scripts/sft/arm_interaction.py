# note (luojiaxuan): 三臂主结果表。外审定的主指标是**固定选择器 q 下的交互效应**,而不是每状态取 max
# 的 ∃-over-15(那只是固定搜索预算下的事后覆盖量,必须如此标注):
#   I = [V_random训(q) − V_random训(recency)] − [V_recency训(q) − V_recency训(recency)]
# q 在所有臂上完全相同、不用被评测的模型去拟合,因此测的是"策略适配",不是"选择器好坏"。
# 候选 6 帧由 (dir, step) 派生的种子决定,跨臂一致,所以任何定义在这 6 帧上的 q 都能在标签集上**离线精确评估**。
# 四个绝对值(null / recency / q / irrelevant)全部报——只靠交互为正而 recency 掉了不算证据。
import argparse, glob, json, os, re, statistics as st, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guiowl_oracle import parse_action, match, shots_for  # noqa: E402

_thumb_cache = {}


def thumbs(d, n):
    if d not in _thumb_cache:
        _thumb_cache[d] = [np.asarray(Image.open(p).convert("L").resize((54, 120)), dtype=np.float32)
                           for p in shots_for(d)[:n]]
    return _thumb_cache[d]


def q_change(rec):
    """固定启发式选择器:在 6 个候选里取"与前一帧视觉变化最大"的两帧。不依赖任何被评测模型。"""
    c = rec["cand_idx"]
    th = thumbs(rec["dir"], max(c) + 2)
    score = {}
    for i in c:
        score[i] = float(np.abs(th[i] - th[i - 1]).mean()) if 0 < i < len(th) else 0.0
    top = sorted(c, key=lambda i: -score[i])[:2]
    return tuple(sorted(c.index(i) for i in top))


def q_spread(rec):
    """固定启发式选择器:取最老与最新的候选帧(最大时间跨度)。"""
    return (0, len(rec["cand_idx"]) - 1)


SELECTORS = {"q_change": q_change, "q_spread": q_spread}


def arm_values(paths, held):
    rows = [json.loads(l) for f in paths for l in open(f)]
    out = {k: [] for k in ("null", "recency", "irrelevant", "q_change", "q_spread", "mean_pair", "exists_pair")}
    for r in rows:
        if r["task"] not in held:
            continue
        ref = parse_action(r["target"])
        cm = {k: match(parse_action(v), ref) for k, v in r["decodes"].items()}
        if ref is None or cm.get("null") is None or cm.get("recency") is None:
            continue
        pairs = [cm[k] or 0 for k in cm if re.fullmatch(r"\d+_\d+", k)]
        if not pairs:
            continue
        out["null"].append(cm["null"] or 0)
        out["recency"].append(cm["recency"] or 0)
        out["irrelevant"].append(cm["irrelevant"] or 0 if cm.get("irrelevant") is not None else float("nan"))
        out["mean_pair"].append(st.mean(pairs))
        out["exists_pair"].append(max(pairs))
        for name, fn in SELECTORS.items():
            a, b = fn(r)
            out[name].append(cm.get(f"{min(a,b)}_{max(a,b)}") or 0)
    return {k: (float(np.nanmean(v)) if v else float("nan")) for k, v in out.items()}, len(out["null"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="name=glob 形式,如 base=labels_base_*.jsonl")
    ap.add_argument("--split", default="/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json")
    ap.add_argument("--baseline-arm", default="recency")
    args = ap.parse_args()
    held = set(json.load(open(args.split))["heldout"])

    vals = {}
    for spec in args.arms:
        name, _, pattern = spec.partition("=")
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"[skip] {name}: no files for {pattern}"); continue
        vals[name], n = arm_values(paths, held)
        v = vals[name]
        print(f"{name:12s} n={n:5d} null={v['null']:.3f} recency={v['recency']:.3f} "
              f"irrelevant={v['irrelevant']:.3f} q_change={v['q_change']:.3f} q_spread={v['q_spread']:.3f} "
              f"| mean_pair={v['mean_pair']:.3f} exists_pair(事后覆盖,非可达)={v['exists_pair']:.3f}")

    base = args.baseline_arm
    if base in vals:
        print(f"\n交互效应 I = [V_arm(q) - V_arm(recency)] - [V_{base}(q) - V_{base}(recency)]")
        for q in SELECTORS:
            b = vals[base][q] - vals[base]["recency"]
            for name, v in vals.items():
                if name == base:
                    continue
                print(f"  {q:9s} {name:12s} I = {(v[q]-v['recency']) - b:+.3f} "
                      f"(该臂 {v[q]-v['recency']:+.3f} vs {base} {b:+.3f})")
        print("\n干扰易感性(irrelevant - null;越负越易被无关历史带偏):")
        for name, v in vals.items():
            print(f"  {name:12s} {v['irrelevant']-v['null']:+.3f}")


if __name__ == "__main__":
    main()
