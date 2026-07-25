"""Contract tests for the sparse-history prompt builder."""
import re, sys, types

# 桩掉 causalcache 依赖,只测渲染逻辑
stub = types.ModuleType("causalcache.policy.gui_owl_official")
stub.OFFICIAL_SYSTEM_PROMPT = "SYS(Action: ... <tool_call>)"
stub.NO_PREVIOUS_ACTION = "No previous action."
sys.modules.setdefault("causalcache", types.ModuleType("causalcache"))
sys.modules.setdefault("causalcache.policy", types.ModuleType("causalcache.policy"))
sys.modules["causalcache.policy.gui_owl_official"] = stub

import importlib.util
spec = importlib.util.spec_from_file_location("sparse", "/private/tmp/claude-501/-Users-luojiaxuan-Documents-CausalCache--claude-worktrees-jolly-greider-bab8b9/84978101-e88d-43e9-8a0a-66a6d8fc5097/scratchpad/sparse.py")
sparse = importlib.util.module_from_spec(spec); spec.loader.exec_module(sparse)

ACTS = [f"do-{i}" for i in range(1, 18)]  # Step1..Step17
def texts(msgs):
    return [p["text"] for m in msgs for p in m["content"] if p.get("type") == "text"]
def order(msgs):
    return [p.get("type") for m in msgs for p in m["content"]]

fails = []
def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond: fails.append(name)

msgs = sparse.build_sparse_history_messages(
    instruction="book a flight", action_texts=ACTS,
    selected_steps=[3, 11, 16], selected_images=["IMG3", "IMG11", "IMG16"],
    current_step=18, current_image="CUR")
blob = "\n".join(texts(msgs))

check("1 历史图按 step 升序", [int(x) for x in re.findall(r"Historical screenshot from Step(\d+)", blob)] == [3, 11, 16])
imgs = [p["image"] for m in msgs for p in m["content"] if p.get("type") == "image"]
check("1b 图像顺序与 step 一致且当前图最后", imgs == ["IMG3", "IMG11", "IMG16", "CUR"])
check("5 当前截图是最后一个 image", order(msgs)[-1] == "image")
mentioned = set(int(x) for x in re.findall(r"Step(\d+):", blob))
check("2 跳过的 step 仍以文本保留(1,2,4..10,12..15,17)", {1,2,4,5,6,7,8,9,10,12,13,14,15,17} <= mentioned)
check("2b 每个 step 文本不重复", all(blob.count(f"Step{i}: do-{i}") == 1 for i in (1,2,5,9,13,17)))
for s in (3, 11, 16):
    check(f"3 Step{s} 图后紧跟其原始动作", f"Action at Step{s}: do-{s}" in blob)
check("4 显式声明 sparse/non-consecutive", "sparse, non-consecutive" in blob and "not necessarily adjacent" in blob)
check("6 沿用官方 system prompt", msgs[0]["content"][0]["text"] == stub.OFFICIAL_SYSTEM_PROMPT)
check("token mask 依据:图数=K+1", sparse.sparse_image_count(msgs) == 4)

m0 = sparse.build_sparse_history_messages(instruction="x", action_texts=ACTS,
    selected_steps=[], selected_images=[], current_step=6, current_image="CUR")
check("K=0 退化为纯文本历史+当前图", sparse.sparse_image_count(m0) == 1 and "Step5: do-5" in "\n".join(texts(m0)))

for bad, kw in ((dict(selected_steps=[11,3], selected_images=["a","b"]), "乱序"),
                (dict(selected_steps=[3,20], selected_images=["a","b"]), "越过当前步"),
                (dict(selected_steps=[3], selected_images=["a","b"]), "图数不匹配")):
    try:
        sparse.build_sparse_history_messages(instruction="x", action_texts=ACTS,
            current_step=18, current_image="C", **bad)
        check(f"fail-closed: {kw}", False)
    except ValueError:
        check(f"fail-closed: {kw}", True)

print("\n%d 项通过, %d 项失败" % (14 - len(fails), len(fails)))
sys.exit(1 if fails else 0)
