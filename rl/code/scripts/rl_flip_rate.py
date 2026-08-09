# note (luojiaxuan): 直接量**逐次非确定性**。allframes 与 labels_all 对
# 同一批 state 的**同一个子集**(recent-2、B=0)各测了一次,两次不一致的比例
# 就是单次评测的翻转率。这个数决定了今天所有结论的可信区间 ——
# 若它有 10%,那么任何小于约 10pp 的单臂差异都可能是噪声。
import json, math
lab = {}
for line in open("/data/oracle/labels_all.jsonl"):
    line = line.strip()
    if not line: continue
    d = json.loads(line)
    if "recent_correct" in d:
        lab[d["dp_id"]] = d
rows = [json.loads(l) for l in open("/data/oracle/allframes.jsonl") if l.strip()]
pairs = [(r, lab[r["dp_id"]]) for r in rows if r["dp_id"] in lab]
print(f"两次都测过的 state:{len(pairs)}")
for arm, k1, k2 in (("recent-2", "recent2", "recent_correct"),
                    ("B=0", "b0", "b0_correct")):
    dis = sum(1 for a, b in pairs if a[k1] != b[k2])
    n = len(pairs)
    # 不一致里两个方向各多少 —— 若严重不对称,说明不是噪声而是口径差异
    up = sum(1 for a, b in pairs if a[k1] and not b[k2])
    dn = sum(1 for a, b in pairs if b[k2] and not a[k1])
    se = 100 * math.sqrt(dis / n * (1 - dis / n) / n) if n else 0
    print(f"  {arm:<9} 不一致 {dis}/{n} = {100*dis/n:.1f}% (±{se:.1f})"
          f";新测对旧测错 {up}、旧测对新测错 {dn}")
print("\n若两个方向大致对称 → 是逐次非确定性;严重偏向一边 → 两次评测口径不同,")
print("那更严重:说明 allframes 的生成配置与枚举不同源,三臂对照本身失效。")
