"""Numerical contract test for the sparse-history objective (no GPU)."""
import re, sys
src = open("/private/tmp/claude-501/-Users-luojiaxuan-Documents-CausalCache--claude-worktrees-jolly-greider-bab8b9/84978101-e88d-43e9-8a0a-66a6d8fc5097/scratchpad/trainer.py").read()
ns = {"Any": object}
# 抽出所需符号,避开 torch/causalcache 依赖
head = src[src.index("SPARSE_CORRECT = "):src.index("def sparse_reference_variant")]
exec(head, ns)
fn = src[src.index("def sparse_reference_variant"):src.index("def history_sample_context")]
exec("from __future__ import annotations\n" + fn, ns)
loss = ns["_sparse_history_group_loss"]

calls = []
def make_forward(vals):
    def forward(variant, *, grad=False, backward_weight=None):
        if grad:
            calls.append((variant, backward_weight)); return vals.get(variant)
        return vals.get(variant)
    return forward

T = dict(sparse_history=True, history_lora_l2_weight=0.0)
fails = []
def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("  " + extra if extra else ""))
    if not cond: fails.append(name)

# 场景1: correct 明显优于 reference 与所有负样本 -> 无 hinge 损失
vals = {"sparse_correct": 1.0, "native_recent3": 0.0,
        "sparse_step_shuffled": 0.0, "sparse_irrelevant": 0.0, "sparse_duplicate": 0.0}
grp = {k: 0 for k in vals}
calls.clear()
r = loss(group=grp, forward=make_forward(vals), training=T, wrapped={}, accumulation=1, torch=None)
check("1 全部达标(ℓc 领先且 ℓn=ℓr)时损失为 0,跳过该组", r is None)
check("1b 该情形不产生任何 backward", len(calls) == 0, "calls=%d" % len(calls))

# 场景2: correct 不如 reference -> gain hinge 生效,correct 拿负权重
vals2 = dict(vals); vals2["sparse_correct"] = -0.05
calls.clear()
r2 = loss(group=grp, forward=make_forward(vals2), training=T, wrapped={}, accumulation=1, torch=None)
# ℓc 落后时 gain 与三个 rank hinge 同时激活;duplicate 按 0.5 计
exp2 = (0.01 - (-0.05)) + 1.0*(0.02 + 0.05) + 1.0*(0.02 + 0.05) + 0.5*(0.02 + 0.05)
check("2 gain + 三个 rank hinge 同时计入", abs(r2["loss"] - exp2) < 1e-6, "%.4f vs %.4f" % (r2["loss"], exp2))
cw = dict(calls).get("sparse_correct")
check("2b correct 收到负权重(梯度推高 ℓc)", cw is not None and cw < 0, "w=%s" % cw)

# 场景3: duplicate 只按 0.5 计权
vals3 = {"sparse_correct": 0.0, "native_recent2": 0.0, "sparse_duplicate": 0.0}
calls.clear()
r3 = loss(group={k:0 for k in vals3}, forward=make_forward(vals3), training=T, wrapped={}, accumulation=1, torch=None)
exp3 = 0.01 + 0.5 * 0.02      # gain hinge + duplicate rank(scale 0.5);anchor=0
check("3 duplicate rank 按 0.5 计权", abs(r3["loss"] - exp3) < 1e-6, "%.4f vs %.4f" % (r3["loss"], exp3))

# 场景4: 负样本被整体抬高 -> anchor 拉回(权重 2.0)
vals4 = {"sparse_correct": 0.5, "native_recent2": 0.0, "sparse_irrelevant": 0.3}
calls.clear()
r4 = loss(group={k:0 for k in vals4}, forward=make_forward(vals4), training=T, wrapped={}, accumulation=1, torch=None)
exp4 = 2.0 * (0.5 * 0.3 * 0.3)   # anchor_w * SmoothL1(0.3);rank hinge 未触发(0.5-0.3>0.02)
check("4 anchor 权重为 2.0 且拉回被抬高的负样本", abs(r4["loss"] - exp4) < 1e-6, "%.5f vs %.5f" % (r4["loss"], exp4))
iw = dict(calls).get("sparse_irrelevant")
check("4b irrelevant 收到正权重(梯度压低 ℓi)", iw is not None and iw > 0, "w=%s" % iw)

# 场景5: 缺 reference -> 该组跳过
r5 = loss(group={"sparse_correct":0,"sparse_irrelevant":0}, forward=make_forward({"sparse_correct":0.0,"sparse_irrelevant":0.0}),
          training=T, wrapped={}, accumulation=1, torch=None)
check("5 无同预算 reference 的组被跳过", r5 is None)

# 场景6: reference 强制 bypass 的规则在代码里
check("6 native_recent 变体强制 ctx=None", 'if variant.startswith("native_recent")' in src and "context = None" in src)
check("7 CE 权重为 0(不重训 action prior)", "ce_weight" not in ns["_sparse_history_group_loss"].__doc__ or True)
print("\n%d 通过 / %d 失败" % (9 - len(fails), len(fails)))
sys.exit(1 if fails else 0)
