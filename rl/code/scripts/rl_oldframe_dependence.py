# note (luojiaxuan): "cross-app 该涨更显著"的机制性检验:MultiApp 的正确子集
# 是否更依赖老帧。老帧定义:帧年龄 = 最新候选帧号 − 该帧号,≥5 为老。
# 两个量,按域分层(仅 winnable):
#   A. 必须老帧率:该态**所有**正确子集都含 ≥1 老帧(recency 类选法必死);
#   B. recent-2 失败且存在含老帧正确子集(selector 能赢 recency 的机会集)。
import collections, json
dom = json.load(open("/data/task_domain.json"))
t2d = {}
for line in open("/data/oracle/agentnet_screening_manifest_ubuntu_v1.jsonl"):
    line = line.strip()
    if line:
        d = json.loads(line); t2d[d["dp_id"]] = dom.get(d.get("task_id"), "?")
stats = collections.defaultdict(lambda: [0, 0, 0])   # 域 -> [winnable, 必老帧, 机会集]
for l in open("/data/oracle/labels_all.jsonl"):
    l = l.strip()
    if not l: continue
    r = json.loads(l)
    if r.get("b0_correct") or not r.get("oracle_correct"): continue
    subs = [(tuple(a["s"]), a["c"]) for a in r.get("all", []) if len(a["s"]) == 2]
    pos = [s for s, c in subs if c]
    if not pos: continue
    cands = sorted({j for s, _ in subs for j in s})
    newest = max(cands)
    def has_old(s): return any(newest - j >= 5 for j in s)
    d = "MultiApp" if t2d.get(r["dp_id"]) == "MultiApp" else "单应用"
    st = stats[d]
    st[0] += 1
    st[1] += int(all(has_old(s) for s in pos))
    st[2] += int((not r["recent_correct"]) and any(has_old(s) for s in pos))
print(f"{'域':<10}{'winnable':>9}{'必须老帧率':>11}{'可赢机会集':>11}")
for d, (n, a, b) in stats.items():
    print(f"{d:<10}{n:>9}{100*a/n:>10.1f}%{100*b/n:>10.1f}%")
