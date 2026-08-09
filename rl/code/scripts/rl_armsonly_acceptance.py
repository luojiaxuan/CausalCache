# note (luojiaxuan): --arms-only 产物的验收判据,四条都必须过:
#  ① 绝不能出现 oracle_correct 字段(写 False 等于断言"没有正确子集",
#     而我们根本没找过 —— 归约端靠字段是否存在决定进不进 oracle 统计);
#  ② mode 必须是 arms_only,便于事后识别这批行的口径;
#  ③ recent_s / random_s 必须在 all 里能查到,否则容差重算取不到臂;
#  ④ 打分子集数应为 2(recent 与 随机;两者相同则为 1),不是几十个 ——
#     这是"确实没枚举 oracle 池"的直接证据。
import json
ok = True
for l in open("/data/oracle/_armstest.jsonl"):
    l = l.strip()
    if not l: continue
    d = json.loads(l)
    subs = {tuple(a["s"]): a for a in d["all"]}
    checks = [
        ("无 oracle 字段", "oracle_correct" not in d and "oracle_subsets" not in d),
        ("mode=arms_only", d.get("mode") == "arms_only"),
        ("recent_s 可查", tuple(d["recent_s"]) in subs),
        ("random_s 可查", tuple(d["random_s"]) in subs),
        ("子集数 ≤2", len(d["all"]) <= 2),
        ("带 pred", all(a.get("p") is not None or not a["c"] for a in d["all"])),
        ("e 标记全 False", all(a["e"] is False for a in d["all"])),
        ("有 gold/b0", "gold" in d and "b0_pred" in d),
    ]
    bad = [n for n, v in checks if not v]
    print(f"{d['dp_id'][:12]} 候选{d['n_candidates']:>3} 打分{len(d['all'])} "
          f"recent={d['recent_correct']} 随机={d['random_correct']} b0={d['b0_correct']}"
          + (f"  ❌ {bad}" if bad else "  ✅"))
    ok &= not bad
print("\n验收:", "全部通过 —— easy 层那条路径可以放心跑" if ok else "❌ 有问题,必须修")
