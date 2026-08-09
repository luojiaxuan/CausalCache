# note (luojiaxuan): **廉价符号特征基线** —— 纯 CPU,不碰模型。
#
# 为什么这个对照能改变判断:我们已经知道两条神经路径(训出来的头 58.3%、
# 未训练的注意力 59.7%)撞在同一个天花板上。但还没问过一个更基本的问题:
# **这个天花板是不是根本就不需要神经网络就能达到?**
# 若"取最近的那帧""该步是 click 而非 type""该步动作文本与目标有词重叠"
# 这类符号特征就能到 59%,那么:
#   ① 说明可迁移信号的**全部内容**就是这些浅层规律,神经特征没带来任何东西;
#   ② selector 该做成一个几行的规则/逻辑回归,而不是 4096 维打分头;
#   ③ 论文的主张要从"学一个 selector"改成"上界很大但只有平凡部分可学"。
# 若符号特征明显低于 59%,则神经特征确实捕捉到了额外的东西,
# 只是不够多 —— 那还值得继续找更好的表示。
#
# 判据与注意力诊断**完全同构**:同一批承载率高/低配对,同一个 AUC 定义。
import json, math, random, re

lab = {}
for line in open("/data/oracle/labels_all.jsonl"):
    line = line.strip()
    if not line: continue
    d = json.loads(line)
    subs = [(tuple(a["s"]), bool(a["c"])) for a in d.get("all", []) if len(a["s"]) == 2]
    if not (any(c for _, c in subs) and any(not c for _, c in subs)): continue
    carry = {}
    for sset, c in subs:
        for j in sset:
            h, t = carry.get(j, (0, 0)); carry[j] = (h + int(c), t + 1)
    rate = {j: h / t for j, (h, t) in carry.items() if t >= 2}
    if len(rate) >= 2:
        hi = max(rate, key=rate.get); lo = min(rate, key=rate.get)
        if rate[hi] - rate[lo] >= 0.2:
            lab[d["dp_id"]] = (hi, lo)
print(f"可判态 {len(lab)}(与注意力诊断同一口径)")

def toks(x):
    return set(re.findall(r"[a-z0-9]+", str(x).lower()))

feats = {
    "越近越好(recency)":      lambda j, ctx: j,
    "越远越好":                lambda j, ctx: -j,
    "该步动作是 click":        lambda j, ctx: int("click" in ctx["act"][j]),
    "该步动作是 type":         lambda j, ctx: int("type" in ctx["act"][j]),
    "动作文本与目标词重叠":     lambda j, ctx: len(ctx["txt"][j] & ctx["goal"]),
    "该步动作文本非空":         lambda j, ctx: int(bool(ctx["txt"][j])),
}
wins = {k: [0, 0] for k in feats}
n = 0
for line in open("/data/oracle/agentnet_screening_manifest_ubuntu_v1.jsonl"):
    line = line.strip()
    if not line: continue
    rec = json.loads(line)
    dp = rec["dp_id"]
    if dp not in lab: continue
    hi, lo = lab[dp]
    hist = rec.get("history", [])
    if max(hi, lo) >= len(hist): continue
    ctx = {"goal": toks(rec.get("instruction", "")),
           "act": {j: str(hist[j].get("raw_code", "")).lower() for j in (hi, lo)},
           "txt": {j: toks(hist[j].get("raw_code", "")) for j in (hi, lo)}}
    for k, f in feats.items():
        a, b = f(hi, ctx), f(lo, ctx)
        if a == b: continue                    # 平局不计,与注意力诊断一致
        w = wins[k]; w[0] += int(a > b); w[1] += 1
    n += 1
print(f"参与统计 {n} 态\n")
def p2(k, m):
    if m == 0: return 1.0
    t = sum(math.comb(m, i) for i in range(min(k, m - k) + 1)) / (2 ** m)
    return min(1.0, 2 * t)
print(f"{'符号特征':<26}{'AUC':>8}{'有效对':>8}{'双侧 p':>10}")
for k, (h, t) in sorted(wins.items(), key=lambda x: -(x[1][0] / max(x[1][1], 1))):
    print(f"{k:<26}{100*h/max(t,1):>7.1f}%{t:>8}{p2(h,t):>10.4f}")
print("\n对照:训练出来的头 58.3% / 未训练注意力 59.7%(n=600)")
print("若某个符号特征逼近或超过 59% → 可迁移信号的全部内容就是这条平凡规律,"
      "\n  神经特征没带来额外东西,selector 该做成规则而非 4096 维打分头。")
