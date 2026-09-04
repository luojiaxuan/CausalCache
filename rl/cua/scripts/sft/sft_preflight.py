# note (luojiaxuan): 三臂发射前的配平核验(纯 CPU,不碰 GPU)。除数据文件外一切必须相同,
# 且三臂的状态键、目标动作、图像张数必须逐条一致——只有 keep_frames 允许不同。
import hashlib, json, sys

paths = {a: f"/data01/jaxan/sglang-omni-rl/sft_{a}_v2.jsonl" for a in ("recency", "older", "random")}
rows = {a: [json.loads(l) for l in open(p)] for a, p in paths.items()}
n = {a: len(v) for a, v in rows.items()}
print("rows:", n)
assert len(set(n.values())) == 1, "曝光量不一致"

base = rows["recency"]
for a in ("older", "random"):
    other = rows[a]
    for i, (x, y) in enumerate(zip(base, other)):
        assert (x["task"], x["step"]) == (y["task"], y["step"]), f"{a} 状态键不一致 @{i}"
        assert x["target"] == y["target"], f"{a} 目标动作不一致 @{i}"
        assert len(x["images"]) == len(y["images"]), f"{a} 图像张数不一致 @{i}"
        assert x["system"] == y["system"] and x["user"] == y["user"], f"{a} 文本历史不一致 @{i}"
    diff = sum(1 for x, y in zip(base, other) if x["keep_frames"] != y["keep_frames"])
    print(f"{a}: 状态键/目标/图数/文本 全部一致;keep_frames 不同的样本 {diff}/{len(other)} = {diff/len(other):.1%}")

# 帧顺序:一律时间序
for a, v in rows.items():
    assert all(r["keep_frames"] == sorted(r["keep_frames"]) for r in v), f"{a} 帧顺序非时间序"
print("帧顺序: 三臂均为时间序")
print("PREFLIGHT_OK")
