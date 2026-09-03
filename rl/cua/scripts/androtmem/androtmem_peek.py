import json, zipfile, itertools, os
D = "/data01/jaxan/androtmem"
z = zipfile.ZipFile(f"{D}/annos.zip"); names = z.namelist()
print("annos.zip entries:", len(names)); print("  ", names[:6])
# 取第一个 json 样本看结构
for n in names:
    if n.endswith(".json"):
        obj = json.loads(z.read(n)); t = obj[0] if isinstance(obj, list) else obj
        print("sample file:", n, "| type:", type(obj).__name__, "| task keys:", list(t.keys()))
        st = t.get("steps", [])
        print("  n_steps:", len(st), "| step keys:", list(st[0].keys()) if st else None)
        for k in ("instruction", "applications", "apps_involved", "task_type", "anchors", "memory", "dependencies"):
            if k in t: print(f"  {k}: {json.dumps(t[k], ensure_ascii=False)[:200]}")
        if st:
            s0 = st[min(3, len(st)-1)]
            for k, v in s0.items(): print(f"    step.{k}: {json.dumps(v, ensure_ascii=False)[:140]}")
        break
# jsonl 首行(若已完整下载)
p = f"{D}/merged_anno_new.jsonl"
with open(p) as f:
    line = f.readline()
try:
    r = json.loads(line); print("jsonl first record keys:", list(r.keys()), "| line bytes:", len(line))
    for k, v in r.items():
        if k != "steps": print(f"  {k}: {json.dumps(v, ensure_ascii=False)[:160]}")
    st = r.get("steps", []); print("  n_steps:", len(st), "| step keys:", list(st[0].keys()) if st else None)
    if st:
        for k, v in st[0].items(): print(f"    step.{k}: {json.dumps(v, ensure_ascii=False)[:120]}")
except Exception as e:
    print("jsonl first line not complete yet:", str(e)[:80], "| bytes so far:", os.path.getsize(p))
