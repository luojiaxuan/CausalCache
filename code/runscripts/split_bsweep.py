#!/usr/bin/env python3
# note (luojiaxuan): 把 B 扫描的 36 任务子集与 18 容器 fleet 各拆成两半,让两个 runner
# 分别打 58900/58901,吃满 GPU4+GPU5。任务按奇偶交错拆分以均衡难度;两个臂用同一份
# 拆分,任务级配对不受影响。哈希字段必须用与 runner 完全一致的算法重算,否则被校验拒绝。
import hashlib, json, sys

R = "/data/mw/runs/bsweep"


def names_sha(values):
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


sub = json.load(open("/data/mw/runs/bsweep_subset.json"))
tasks = sub["tasks"]
assert len(tasks) == 36, len(tasks)
assert names_sha(tasks) == sub["task_names_sha256"], "源子集哈希自检失败"

halves = {"a": tasks[0::2], "b": tasks[1::2]}
for tag, tk in halves.items():
    out = dict(sub)
    out["tasks"] = tk
    out["task_count"] = len(tk)
    out["task_names_sha256"] = names_sha(tk)
    out["shard_index"] = 0 if tag == "a" else 1
    out["shard_count"] = 2
    out["shard_assigned_count"] = len(tk)
    # full_roster_sha256 / full_roster_count 保持不变——它校验的是全 117 名单
    json.dump(out, open(f"{R}/subset-{tag}.json", "w"), indent=1, sort_keys=True)
    print(f"subset-{tag}: {len(tk)} tasks sha={out['task_names_sha256'][:16]}")

fleet = json.load(open(f"{R}/fleet.json"))
cont = fleet["containers"]
assert len(cont) == 18, len(cont)
for tag, sl in (("a", cont[:9]), ("b", cont[9:])):
    out = dict(fleet)
    out["containers"] = sl
    json.dump(out, open(f"{R}/fleet-{tag}.json", "w"), indent=1, sort_keys=True)
    print(f"fleet-{tag}: {len(sl)} envs {[c['name'] for c in sl]}")

# 交叉检查:两半互不重叠且并集等于原集
assert set(halves["a"]) | set(halves["b"]) == set(tasks)
assert not (set(halves["a"]) & set(halves["b"]))
assert not ({c["name"] for c in cont[:9]} & {c["name"] for c in cont[9:]})
print("OK: 任务与容器两半均不重叠,并集完整")
