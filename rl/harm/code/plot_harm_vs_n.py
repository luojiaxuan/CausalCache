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
fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
summary = {}
for ax, proto, title in ((axes[0], "deploy", "deployed layout (interleaved turns)"), (axes[1], "text", "labeller layout: images first + all conclusions"), (axes[2], "notext", "labeller layout, text trace removed")):
    rows = load(f"{H}/harm_vs_n_base_{proto}.jsonl")
    if not rows: ax.set_title(title + " (no data)"); continue
    suf = "_deploy" if proto == "deploy" else ""
    S = {k.replace(suf, ""): v for k, v in series(rows, [f"rec{n}{suf}" for n in Ns] + [f"irr{n}{suf}" for n in Ns if n > 0]).items()}; summary[proto] = S
    for kind, lab, c in (("rec", "real most-recent N frames", "tab:blue"), ("irr", "N frames from another task", "tab:gray")):
        xs = [n for n in Ns if f"{kind}{n}" in S or (kind == "irr" and n == 0 and "rec0" in S)]
        ys, lo, hi = [], [], []
        for n in xs:
            k = "rec0" if (kind == "irr" and n == 0) else f"{kind}{n}"; m, l, h, _ = S[k]; ys.append(100*m); lo.append(100*(m-l)); hi.append(100*(h-m))
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=lab, color=c)
    ax.set_xlabel("number of history screenshots N"); ax.set_ylabel("action reproduction (%)"); ax.set_title(f"frozen GUI-Owl-1.5-8B · {title} · n={len(rows)}", fontsize=9); ax.grid(alpha=.3); ax.legend()
ax = axes[3]; rows = load(f"{H}/harm_vs_n_base_text.jsonl")
if rows:
    S = series(rows, ["rec0", "rec2", "rec2_mark", "rec2_blank", "irr2", "rec4", "rec4_mark", "rec4_blank", "irr4"]); summary["interv"] = S
    order = ["rec0", "rec2", "rec2_mark", "rec2_blank", "irr2", "rec4", "rec4_mark", "rec4_blank", "irr4"]
    labels = ["N=0", "N=2 real", "N=2 +PAST tag", "N=2 gray", "N=2 irrelevant", "N=4 real", "N=4 +PAST tag", "N=4 gray", "N=4 irrelevant"]
    xs = list(range(len(order))); ys = [100*S[k][0] if k in S else 0 for k in order]
    err = [[100*(S[k][0]-S[k][1]) if k in S else 0 for k in order], [100*(S[k][2]-S[k][0]) if k in S else 0 for k in order]]
    cols = ["k", "tab:blue", "tab:green", "tab:orange", "tab:gray", "tab:blue", "tab:green", "tab:orange", "tab:gray"]
    ax.bar(xs, ys, yerr=err, capsize=3, color=cols, alpha=.85); ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("action reproduction (%)"); ax.set_title("interventions (labeller layout): PAST tag vs same-token gray image", fontsize=9); ax.grid(axis="y", alpha=.3)
    if "rec0" in S: ax.axhline(100*S["rec0"][0], ls="--", color="k", lw=.8)
plt.tight_layout(); plt.savefig(f"{H}/fig1_harm_vs_n.png", dpi=150); print("saved", f"{H}/fig1_harm_vs_n.png")
for proto, S in summary.items():
    print(f"\n[{proto}]"); [print(f"  {k:12s} {100*m:5.1f} [{100*l:5.1f}, {100*h:5.1f}] n={n}") for k, (m, l, h, n) in S.items()]
