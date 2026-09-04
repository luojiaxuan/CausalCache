# note (luojiaxuan): GUI-Owl 标签集的归约。同一份脚本用于三处、口径完全一致:冻结底座、SFT-recency 臂、
# SFT-random 臂——三者的差就是论文第二章的主结果。除 ∃-over-15-pairs(带赢家诅咒)外,同时报两个
# 不受该诅咒影响的量:**上下文不敏感步占比**(23 个上下文下动作等价类只有一种)与 **recency 错时的可救率**。
# 训练/未见模板按 mw_split_v1.json 切分;主张只看未见模板那一栏。
import collections, json, statistics as st, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guiowl_oracle import parse_action, match  # noqa: E402

SPLIT = os.environ.get("CC_SPLIT", "/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json")
HELD = set(json.load(open(SPLIT))["heldout"]) if os.path.exists(SPLIT) else set()


def canon(a):
    p = parse_action(a)
    if p is None:
        return None
    c = p["coord"] if isinstance(p["coord"], (list, tuple)) and len(p["coord"]) == 2 else (0, 0)
    return (p["action"], round(c[0] / 60), round(c[1] / 60), p["text"][:40])


def summarize(recs, label):
    rec, null, mean_pair, best_pair, best_single, insens = [], [], [], [], [], []
    rescue = rec_wrong = 0
    for r in recs:
        ref = parse_action(r["target"])
        cm = {k: match(parse_action(v), ref) for k, v in r["decodes"].items()}
        if ref is None or cm.get("recency") is None or cm.get("null") is None:
            continue
        pairs = [cm[k] or 0 for k in cm if "_" in k and not k.startswith("s")]
        singles = [cm[f"s{i}"] or 0 for i in range(6) if cm.get(f"s{i}") is not None]
        if not pairs:
            continue
        rec.append(cm["recency"] or 0); null.append(cm["null"] or 0)
        mean_pair.append(st.mean(pairs)); best_pair.append(max(pairs))
        best_single.append(max(singles) if singles else 0)
        insens.append(1 if len({canon(v) for v in r["decodes"].values()} - {None}) <= 1 else 0)
        if not cm["recency"]:
            rec_wrong += 1; rescue += max(pairs)
    n = len(rec)
    if not n:
        print(f"[{label}] no states"); return
    print(f"[{label}] n={n} null={st.mean(null):.3f} recency={st.mean(rec):.3f} random_pair={st.mean(mean_pair):.3f} "
          f"oracle_pair={st.mean(best_pair):.3f} oracle_single={st.mean(best_single):.3f} "
          f"| oracle-recency={st.mean(best_pair)-st.mean(rec):+.3f} "
          f"context_insensitive={st.mean(insens):.1%} rescue={rescue}/{rec_wrong}")


recs = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
summarize(recs, "all")
if HELD:
    summarize([r for r in recs if r["task"] in HELD], "heldout")
    summarize([r for r in recs if r["task"] not in HELD], "train")
by = collections.Counter(r["root"] for r in recs)
print("states by source arm:", dict(by.most_common()))
