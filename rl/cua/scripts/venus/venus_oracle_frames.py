# note (luojiaxuan): G2 之后的"赢在哪几帧"分析(外审攻击 #2 的诊断输入):在 recency 错、某对能救的状态上,
# 统计救回对里较老那帧的帧龄分布、是否含最近帧、以及"视觉变化最大的历史帧"(change2 启发式)是否落在救回对里。
# 若一条无参数规则就能覆盖大半救回,selector 的失败更像捷径/表征问题而非容量问题。
import glob, json, os, re, statistics as st, sys
import numpy as np
from PIL import Image
sys.path.insert(0, "/data01/jaxan")
from venus_oracle_verdict import parse, match  # noqa: E402

def thumbs(shots):
    return [np.asarray(Image.open(p).convert("L").resize((54, 120)), dtype=np.float32) for p in shots]

recs = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
ages, has_recent, change_hit, n_rescue = [], 0, 0, 0
for r in recs:
    ref = parse(r["target"])
    cm = {k: match(parse(v), ref) for k, v in r["decodes"].items()}
    if cm.get("recency") is None or cm["recency"]:
        continue
    wins = [k for k in cm if "_" in k and not k.startswith("s") and cm[k]]
    if not wins:
        continue
    n_rescue += 1
    k = r["step"]; c = r["cand_idx"]
    shots = sorted(glob.glob(os.path.join(r["dir"], "screenshots", "*.png")),
                   key=lambda p: int(re.search(r"-(\d+)\.png$", p).group(1)))
    th = thumbs(shots[:k - 1])
    scores = [0.0] + [float(np.abs(th[i] - th[i - 1]).mean()) for i in range(1, len(th) - 1)]
    best_change = int(np.argmax(scores)) if scores else 0
    ages.append(min((k - 1) - min(c[int(a)], c[int(b)]) for a, b in (w.split("_") for w in wins)))
    has_recent += any(max(c[int(a)], c[int(b)]) == k - 2 for a, b in (w.split("_") for w in wins))
    change_hit += any(best_change in (c[int(a)], c[int(b)]) for a, b in (w.split("_") for w in wins))
print(f"rescue states={n_rescue}")
if n_rescue:
    print(f"older-frame age in rescuing pair: median={st.median(ages)} >=5:{sum(a>=5 for a in ages)/n_rescue:.2f} >=10:{sum(a>=10 for a in ages)/n_rescue:.2f}")
    print(f"some rescuing pair contains most-recent frame: {has_recent/n_rescue:.2f}")
    print(f"max-visual-change frame inside a rescuing pair: {change_hit/n_rescue:.2f}")
