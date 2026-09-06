# note (luojiaxuan): 部署布局下 N≥4 的骤降是上下文长度效应还是构造问题?按历史长度分层看,并检查 N≥4 时输出是否异常(空/截断/格式错)。
import json, statistics as st, collections
rows = [json.loads(l) for l in open("/data01/jaxan/harm/harm_vs_n_base_deploy.jsonl")]
keys = ["rec0", "rec1", "rec2", "rec3", "rec4", "rec6", "irr2", "irr4"]
print(f"  {'历史长度 k-1':12s} {'n':>4s} " + " ".join(f"{k:>6s}" for k in keys))
for lo, hi in ((5, 9), (10, 19), (20, 29), (30, 50)):
    rs = [r for r in rows if lo <= r["step"] - 1 <= hi]
    if rs: print(f"  {lo:2d}–{hi:2d}         {len(rs):4d} " + " ".join(f"{100*st.mean([bool(r['match'][k+'_deploy']) for r in rs]):6.1f}" for k in keys))
print("\n  N=4 时输出形态(前 6 个 rec2 对、rec4 错的状态):")
n = 0
for r in rows:
    if r["match"].get("rec2_deploy") and not r["match"].get("rec4_deploy"):
        d = r["decodes"]["rec4_deploy"]; print(f"   k={r['step']:2d} len={len(d):4d} has_tool_call={'<tool_call>' in d} head={d[:90]!r}"); n += 1
        if n >= 6: break
c = collections.Counter()
for r in rows:
    for k in ("rec2_deploy", "rec4_deploy", "rec6_deploy"):
        d = r["decodes"][k]; c[(k, "empty" if not d.strip() else ("no_tool_call" if "<tool_call>" not in d else "ok"))] += 1
print("\n  输出形态计数:", dict(c))
