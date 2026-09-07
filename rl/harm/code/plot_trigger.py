# note (luojiaxuan): 图 2——机制 A 的触发物:终止率(总 terminate %)按条件排布。左:剂量曲线(指令之后 M 张灰图 vs 等 token 长文本);
# 右:同为 5 张图的各种放法。数据来自部署布局的各批解码(harm_vs_n_base_deploy / format* / reply_pattern* / dose_base)。
import json, os, re, random, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = "/data01/jaxan/harm"
files = ["harm_vs_n_base_deploy.jsonl", "format_base.jsonl", "format2_base.jsonl", "term_interv_base.jsonl", "goal_interv_base.jsonl", "reply_pattern_base.jsonl", "reply_pattern2_base.jsonl", "dose_base.jsonl", "dose2_base.jsonl", "dose3_base.jsonl", "dose4_base.jsonl", "dose5_base.jsonl", "dose6_base.jsonl", "dose7_base.jsonl"]
recs = {}
for f in files:
    p = f"{H}/{f}"
    if not os.path.exists(p): print("missing", f); continue
    for l in open(p):
        r = json.loads(l); k = f"{r['dir']}|{r['step']}"; recs.setdefault(k, {})
        for sp, m in r["match"].items():
            if m is not None: recs[k][sp] = (bool(m), "terminate" in (r["decodes"].get(sp) or "").lower())
def stat(sp):
    v = [recs[k][sp] for k in recs if sp in recs[k]]
    if not v: return None
    term = [t for _, t in v]; rep = [m for m, _ in v]
    rng = random.Random(0); n = len(v); ms = sorted(st.mean(rng.choices(term, k=n)) for _ in range(1000))
    return 100*st.mean(term), 100*ms[25], 100*ms[975], 100*st.mean(rep), n
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))
# 左:剂量曲线
for kind, lab, c in (("grayturnin", "M gray images AFTER the instruction", "tab:orange"), ("graybefore", "same M gray images BEFORE the instruction", "tab:purple"), ("longtextturnin", "M text blocks after the instruction (~1 screenshot of tokens each)", "tab:green"), ("graystackturnin", "M gray screenshots stacked into ONE image block, after the instruction", "tab:brown")):
    xs, ys, lo, hi = [], [], [], []
    for m in (1, 2, 3, 4):
        r = stat(f"{kind}{m}")
        if r: xs.append(m); ys.append(r[0]); lo.append(r[0]-r[1]); hi.append(r[2]-r[0])
    if xs: ax1.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=lab, color=c)
r0 = stat("rec2_deploy"); r4 = stat("rec4_deploy")
if r4: ax1.axhline(r4[0], ls="--", color="tab:blue", lw=.9, label=f"rec4 (4 real turns) = {r4[0]:.1f}%")
ax1.set_xlabel("dose M (screenshot-equivalents of content-free filler)"); ax1.set_ylabel("terminate rate (%)"); ax1.set_xticks([1, 2, 3, 4])
ax1.set_title("dose x placement: only image blocks AFTER the instruction trigger termination", fontsize=9); ax1.grid(alpha=.3); ax1.legend(fontsize=7)
# 右:5 张图的各种放法
conds = [("rec2_deploy", "rec2 (3 imgs)"), ("rec3_deploy", "rec3 (4 imgs)"), ("rec4_deploy", "rec4 (5 imgs)"), ("irr4_deploy", "irr4 (5 imgs, other task)"),
         ("hybrid:judge_glm46v|direct|six", "2 extra imgs inside instruction msg"), ("hybridturnin:judge_glm46v|direct|six", "2 ref turns after instruction"),
         ("hybridturnin_gray:judge_glm46v|direct|six", "2 gray ref turns after instruction"), ("hybridturnin_text:judge_glm46v|direct|six", "2 text-only ref turns after instruction"),
         ("hybridlast:judge_glm46v|direct|six", "2 extra imgs next to current frame"), ("hybridturn:judge_glm46v|direct|six", "2 ref turns BEFORE instruction"),
         ("graytinyturnin2", "2 ref turns, each a 0.2-screenshot gray (~500 tokens)"), ("graypairturnin2", "1 ref turn holding 2 gray imgs"), ("graypairturnin3", "1 ref turn holding 3 gray imgs"),
         ("rec4_deploy_noted", "rec4, early replies -> 'Noted'"), ("rec4_deploy_short", "rec4, replies emptied")]
labels, ys, lo, hi, reps = [], [], [], [], []
for sp, lab in conds:
    r = stat(sp)
    if r: labels.append(lab); ys.append(r[0]); lo.append(r[0]-r[1]); hi.append(r[2]-r[0]); reps.append(r[3])
xs = list(range(len(labels)))
ax2.bar(xs, ys, yerr=[lo, hi], capsize=3, color=["tab:red" if y > 10 else "tab:blue" for y in ys], alpha=.85)
for x, y, rp in zip(xs, ys, reps): ax2.text(x, y + 1, f"{rp:.0f}", ha="center", fontsize=7, color="gray")
ax2.set_xticks(xs); ax2.set_xticklabels(labels, rotation=40, ha="right", fontsize=7); ax2.set_ylabel("terminate rate (%)")
ax2.set_title("placement decides it (gray number = action reproduction %)", fontsize=9); ax2.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(f"{H}/fig2_trigger.png", dpi=150); print("saved", f"{H}/fig2_trigger.png")
for sp, lab in conds + [(f"{k}{m}", f"{k}{m}") for k in ("grayturnin", "graybefore", "longtextturnin", "graystackturnin") for m in (1, 2, 3, 4)]:
    r = stat(sp)
    if r: print(f"  {sp:44s} term {r[0]:5.1f} [{r[1]:5.1f},{r[2]:5.1f}]  rep {r[3]:5.1f}  n={r[4]}")
