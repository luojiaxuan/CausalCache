# note (luojiaxuan): 用 AgentNet 源文件的 domain 标注(MultiApp = 显式 cross-app)
# 给全部枚举态分层,回答两个问题:
#   ① 我们的 easy/winnable/hopeless 在 MultiApp 与单应用上怎么分布;
#   ② **oracle 头寸是否集中在 MultiApp** —— 若是,"cross-app 涨更显著"
#     就有标签侧的先验支撑,selector 的验收分层直接用真标签。
import collections, json
dom = json.load(open("/data/task_domain.json"))
t2d = {}
for line in open("/data/oracle/agentnet_screening_manifest_ubuntu_v1.jsonl"):
    line = line.strip()
    if line:
        d = json.loads(line)
        t2d[d["dp_id"]] = dom.get(d.get("task_id"), "?")
rows = [json.loads(l) for l in open("/data/oracle/labels_all.jsonl") if l.strip()]
def grp(r):
    if r["b0_correct"]: return "easy"
    if "oracle_correct" not in r: return "?"
    return "winnable" if r["oracle_correct"] else "hopeless"
cnt = collections.Counter()
for r in rows:
    d = "MultiApp" if t2d.get(r["dp_id"]) == "MultiApp" else "单应用"
    cnt[(d, grp(r))] += 1
print("分层 × 域 计数:")
for k in sorted(cnt): print(f"  {k}: {cnt[k]}")
print()
for d in ("MultiApp", "单应用"):
    sel = [r for r in rows if ("MultiApp" if t2d.get(r["dp_id"])=="MultiApp" else "单应用")==d
           and "oracle_correct" in r and not r["b0_correct"]]
    n = len(sel)
    if not n: continue
    o = sum(r["oracle_correct"] for r in sel); rc = sum(r["recent_correct"] for r in sel)
    print(f"{d} 困难态 n={n}: oracle-2 {100*o/n:.1f}%  recent-2 {100*rc/n:.1f}%  头寸 {100*(o-rc)/n:+.1f}pp")
