# note (luojiaxuan):AndroTMem 覆盖率:公开 imgs.zip 只有 11.6k 张图(34k 步),需要知道
# "记忆敏感步(context_use/entity_binding 且时滞≥5)且当前图 + gold 来源图都在"的可用量,
# 以及这些步的 GT 是否带 bbox(tap 判分依赖 bbox)。
import json, zipfile, os, collections
D = "/data01/jaxan/androtmem"; IMG = f"{D}/imgs_raw/imgs"
have = set(os.listdir(IMG))
z = zipfile.ZipFile(f"{D}/annos.zip"); tasks = []
for n in z.namelist():
    if n.endswith(".json"):
        o = json.loads(z.read(n)); tasks.extend(o if isinstance(o, list) else [o])
MEM = {"context_use", "entity_binding"}
tot = img = 0; sens = sens_img = sens_img_gold = sens_full = 0; gt = collections.Counter(); bbox_ok = 0; tasks_full = 0
lag_ok = collections.Counter()
for t in tasks:
    steps = t["steps"]; idx = {s["step_index"]: i for i, s in enumerate(steps)}
    present = [s["image_name"] in have for s in steps]
    tot += len(steps); img += sum(present); tasks_full += all(present)
    for i, s in enumerate(steps):
        links = (s.get("extra_info") or {}).get("links") or []
        gold = [idx.get(l["source"]) for l in links if l.get("relation") in MEM]
        gold = [j for j in gold if j is not None and i - j >= 5]
        if not gold: continue
        sens += 1; sens_img += present[i]
        gold_have = [j for j in gold if present[j]]
        if present[i] and gold_have:
            sens_img_gold += 1; af = s.get("actionForm") or {}; a = af.get("action"); gt[a] += 1
            if a in ("tap", "long_press"): bbox_ok += bool(af.get("bbox"))
            lag_ok[min(i - max(gold_have), 20)] += 1
print(f"steps={tot} with_image={img} ({img/tot*100:.1f}%)  tasks_all_images={tasks_full}/{len(tasks)}")
print(f"sensitive steps (mem-link lag>=5)={sens}  cur_img={sens_img}  cur+gold_img={sens_img_gold}")
print(f"  GT action types among usable: {gt.most_common(8)}")
print(f"  tap/long_press with bbox: {bbox_ok}/{gt['tap']+gt['long_press']}")
print(f"  lag(to nearest gold with image) hist: {dict(sorted(lag_ok.items()))}")
