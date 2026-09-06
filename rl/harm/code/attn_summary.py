# note (luojiaxuan): 注意力探针汇总——首个生成位置的注意力质量分配(heads 平均)随 N 的变化,按层报:当前屏图像 / 历史帧图像合计 / 文本。
import json, statistics as st, collections
rows = [json.loads(l) for l in open("/data01/jaxan/harm/attn_probe_base.jsonl")]
specs = ["rec0_deploy", "rec2_deploy", "rec4_deploy", "rec6_deploy"]
layers = sorted({int(k) for r in rows for k in r["layers"]})
print(f"states={len({(r['dir'],r['step']) for r in rows})}  seq 长度均值: " + ", ".join(f"{sp[:4]}={st.mean([r['seq'] for r in rows if r['spec']==sp]):.0f}" for sp in specs))
print(f"\n  {'层':>3s} | " + " | ".join(f"{sp[:4]:>22s}" for sp in specs) + "      (每格:当前屏 / 历史帧合计 / 文本,注意力质量 %)")
for L in layers:
    row = f"  {L:3d} | "
    for sp in specs:
        rs = [r["layers"][str(L)] for r in rows if r["spec"] == sp and str(L) in r["layers"]]
        if not rs: row += f"{'—':>22s} | "; continue
        cur = 100 * st.mean(x["current_image"] for x in rs); hist = 100 * st.mean(sum(x["history_images"]) for x in rs); txt = 100 * st.mean(x["text"] for x in rs)
        row += f"{cur:5.1f} / {hist:5.1f} / {txt:5.1f}".rjust(22) + " | "
    print(row)
print("\n  当前屏图像注意力占比(全层平均)随 N:")
for sp in specs:
    v = [x["current_image"] for r in rows if r["spec"] == sp for x in r["layers"].values()]
    h = [sum(x["history_images"]) for r in rows if r["spec"] == sp for x in r["layers"].values()]
    print(f"    {sp:12s} 当前屏 {100*st.mean(v):5.1f}%   历史帧 {100*st.mean(h):5.1f}%   历史帧均摊每帧 {100*st.mean(h)/max(int(sp[3])-0,1) if sp!='rec0_deploy' else 0:5.1f}%")
# 逐帧:N=4 时 4 张历史帧各占多少(按时间序,最后一张最近)
for sp in ("rec4_deploy", "rec6_deploy"):
    per = collections.defaultdict(list)
    for r in rows:
        if r["spec"] != sp: continue
        for x in r["layers"].values():
            for i, v in enumerate(x["history_images"]): per[i].append(v)
    print(f"    {sp}: 各历史帧(旧→新)注意力 % = " + ", ".join(f"{100*st.mean(per[i]):.1f}" for i in sorted(per)))

print("\n  文本注意力按轮次拆分(全层平均,%):system / 首条 user 文本 / assistant 回复合计(旧→新) / 其后 user 文本")
for sp in specs:
    rs = [x for r in rows if r["spec"] == sp for x in r["layers"].values() if "assistant_texts" in x]
    if not rs: continue
    na = max(len(x["assistant_texts"]) for x in rs)
    asst = [100 * st.mean(x["assistant_texts"][i] if i < len(x["assistant_texts"]) else 0.0 for x in rs) for i in range(na)]
    print(f"    {sp:12s} sys {100*st.mean(x['system_text'] for x in rs):5.1f} | user0 {100*st.mean(x['first_user_text'] for x in rs):5.1f} | asst " + ", ".join(f"{v:.1f}" for v in asst) + f" | later_user {100*st.mean(sum(x['later_user_texts']) for x in rs):5.1f}")
