# note (luojiaxuan): AndroTMem-Bench 统计:任务/步数、links(记忆因果边)的覆盖率、时滞分布、
# 关键边占比、关系类型——决定它作为 Stage I 训练/oracle 上界数据的价值。
import json, zipfile, collections
z = zipfile.ZipFile("/data01/jaxan/androtmem/annos.zip")
tasks = []
for n in z.namelist():
    if n.endswith(".json"):
        obj = json.loads(z.read(n)); tasks.extend(obj if isinstance(obj, list) else [obj])
n_steps = sum(len(t["steps"]) for t in tasks)
steps_with_links = 0; lags = []; crit = 0; nlinks = 0; rel = collections.Counter(); status_pref = collections.Counter(); acts = collections.Counter()
far5 = far10 = 0
for t in tasks:
    idx = {s["step_index"]: i for i, s in enumerate(t["steps"])}
    for i, s in enumerate(t["steps"]):
        acts[(s.get("actionForm") or {}).get("action", "?")] += 1
        ei = s.get("extra_info") or {}
        for st in ei.get("status") or []:
            c = st.get("content", ""); status_pref[c.split("]")[0] + "]" if c.startswith("[") else "other"] += 1
        links = ei.get("links") or []
        if links: steps_with_links += 1
        for l in links:
            nlinks += 1; rel[l.get("relation", "?")] += 1; crit += bool(l.get("is_critical"))
            src = l.get("source"); j = idx.get(src, idx.get(str(src)))
            if j is not None:
                lag = i - j; lags.append(lag); far5 += lag >= 5; far10 += lag >= 10
print(f"tasks={len(tasks)} steps={n_steps} avg_steps={n_steps/len(tasks):.1f}")
print(f"steps_with_links={steps_with_links} ({steps_with_links/n_steps*100:.1f}%)  links={nlinks} critical={crit} ({crit/max(nlinks,1)*100:.1f}%)")
print(f"link lag: n={len(lags)} mean={sum(lags)/max(len(lags),1):.1f} >=5: {far5/max(len(lags),1)*100:.1f}% >=10: {far10/max(len(lags),1)*100:.1f}%")
h = collections.Counter(min(l, 20) for l in lags); print("lag hist (20=>=20):", {k: h[k] for k in sorted(h)})
print("relations:", rel.most_common(8))
print("status prefixes:", status_pref.most_common(8))
print("actions:", acts.most_common(10))
apps = collections.Counter(a for t in tasks for a in t.get("apps_involved", [])); print("apps top:", apps.most_common(12), "| n_apps=", len(apps))
