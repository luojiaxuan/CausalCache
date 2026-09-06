# note (luojiaxuan): 过早终止与轨迹相对位置的关系:按 k/len(traj) 分箱,看 rec4_deploy 输出 terminate 的比例;
# 同时看 rec2_deploy(几乎不终止)作对照,以及 N=4 下"错误里 terminate"是否随位置变化。
import json, os, sys, statistics as st
sys.path.insert(0, "/data01/jaxan")
from guiowl_oracle import parse_action
rows = [json.loads(l) for l in open("/data01/jaxan/harm/harm_vs_n_base_deploy.jsonl")]
L = {}
for r in rows:
    d = r["dir"]
    if d not in L:
        L[d] = len(list(json.load(open(os.path.join(d, "traj.json"))).values())[0]["traj"])
def is_term(txt): return (parse_action(txt) or {}).get("action") in ("terminate", "finished")
bins = [(0, .25), (.25, .5), (.5, .75), (.75, .9), (.9, 1.01)]
print(f"  {'相对位置 k/len':14s} {'n':>4s} {'rec2 复现':>9s} {'rec4 复现':>9s} {'rec4 输出 terminate':>18s} {'rec6 输出 terminate':>18s}")
for lo, hi in bins:
    rs = [r for r in rows if lo <= r["step"] / L[r["dir"]] < hi]
    if not rs: continue
    print(f"  {lo:.2f}–{hi:.2f}      {len(rs):4d} {100*st.mean([bool(r['match']['rec2_deploy']) for r in rs]):9.1f} {100*st.mean([bool(r['match']['rec4_deploy']) for r in rs]):9.1f} "
          f"{100*st.mean([is_term(r['decodes']['rec4_deploy']) for r in rs]):18.1f} {100*st.mean([is_term(r['decodes']['rec6_deploy']) for r in rs]):18.1f}")
print("\n  终止率与相对位置的相关(Spearman 近似:按分箱单调?)——若前 25% 的步也大量终止,则不是'快做完'的合理判断。")
# 还有一个直接检验:rec4 时 text_cnt(压成文本的步数)与终止的关系——文本历史越长越终止?
print(f"\n  {'历史长度 k-1':12s} {'n':>4s} {'rec4 terminate%':>16s}")
for lo, hi in ((5, 9), (10, 19), (20, 29), (30, 50)):
    rs = [r for r in rows if lo <= r["step"] - 1 <= hi]
    if rs: print(f"  {lo:2d}–{hi:2d}         {len(rs):4d} {100*st.mean([is_term(r['decodes']['rec4_deploy']) for r in rs]):16.1f}")
