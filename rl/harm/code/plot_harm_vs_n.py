# note (luojiaxuan): history-harm 第一张图——动作复现率 vs 历史图数 N(recN 真实最近帧 / irrN 无关帧),两种协议两个面板;
# 第三面板:干预(rec2/rec4 的 PAST 标记版与灰图版)。误差棒 = 按状态 bootstrap 95% CI。输入 harm_vs_n_base_{text,notext}.jsonl。
import json, os, random, statistics as st, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = "/data01/jaxan/harm"
def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def ci(vals, B=2000, seed=0):
    r = random.Random(seed); n = len(vals); ms = sorted(st.mean(r.choices(vals, k=n)) for _ in range(B)); return ms[int(.025*B)], ms[int(.975*B)]
def series(rows, specs):
    out = {}
    for sp in specs:
        v = [bool(r["match"][sp]) for r in rows if r["match"].get(sp) is not None]
        if v: lo, hi = ci(v); out[sp] = (st.mean(v), lo, hi, len(v))
    return out
Ns = [0, 1, 2, 3, 4, 6]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
summary = {}
for ax, proto, title in ((axes[0], "text", "保留文本轨迹(部署条件)"), (axes[1], "notext", "去掉文本轨迹")):
    rows = load(f"{H}/harm_vs_n_base_{proto}.jsonl")
    if not rows: ax.set_title(title + " (无数据)"); continue
    S = series(rows, [f"rec{n}" for n in Ns] + [f"irr{n}" for n in Ns if n > 0]); summary[proto] = S
    for kind, lab, c in (("rec", "真实最近 N 帧", "tab:blue"), ("irr", "无关任务的 N 帧", "tab:gray")):
        xs = [n for n in Ns if f"{kind}{n}" in S or (kind == "irr" and n == 0 and "rec0" in S)]
        ys, lo, hi = [], [], []
        for n in xs:
            k = "rec0" if (kind == "irr" and n == 0) else f"{kind}{n}"; m, l, h, _ = S[k]; ys.append(100*m); lo.append(100*(m-l)); hi.append(100*(h-m))
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=lab, color=c)
    ax.set_xlabel("历史截图数 N"); ax.set_ylabel("动作复现率 (%)"); ax.set_title(f"GUI-Owl-1.5-8B 冻结 · {title} · n={len(rows)}"); ax.grid(alpha=.3); ax.legend()
ax = axes[2]; rows = load(f"{H}/harm_vs_n_base_text.jsonl")
if rows:
    S = series(rows, ["rec0", "rec2", "rec2_mark", "rec2_blank", "irr2", "rec4", "rec4_mark", "rec4_blank", "irr4"]); summary["interv"] = S
    order = ["rec0", "rec2", "rec2_mark", "rec2_blank", "irr2", "rec4", "rec4_mark", "rec4_blank", "irr4"]
    labels = ["N=0", "N=2 真实", "N=2 +PAST 标记", "N=2 灰图", "N=2 无关", "N=4 真实", "N=4 +PAST 标记", "N=4 灰图", "N=4 无关"]
    xs = list(range(len(order))); ys = [100*S[k][0] if k in S else 0 for k in order]
    err = [[100*(S[k][0]-S[k][1]) if k in S else 0 for k in order], [100*(S[k][2]-S[k][0]) if k in S else 0 for k in order]]
    cols = ["k", "tab:blue", "tab:green", "tab:orange", "tab:gray", "tab:blue", "tab:green", "tab:orange", "tab:gray"]
    ax.bar(xs, ys, yerr=err, capsize=3, color=cols, alpha=.85); ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("动作复现率 (%)"); ax.set_title("干预:时序标记 vs 同 token 数灰图(保留文本轨迹)"); ax.grid(axis="y", alpha=.3)
    if "rec0" in S: ax.axhline(100*S["rec0"][0], ls="--", color="k", lw=.8)
plt.rcParams["font.family"] = ["Noto Sans CJK SC", "PingFang SC", "DejaVu Sans"]
plt.tight_layout(); plt.savefig(f"{H}/fig1_harm_vs_n.png", dpi=150); print("saved", f"{H}/fig1_harm_vs_n.png")
for proto, S in summary.items():
    print(f"\n[{proto}]"); [print(f"  {k:12s} {100*m:5.1f} [{100*l:5.1f}, {100*h:5.1f}] n={n}") for k, (m, l, h, n) in S.items()]
