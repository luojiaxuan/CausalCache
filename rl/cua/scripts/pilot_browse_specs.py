# note (luojiaxuan): 延后揭示探针的规格构建。两段式:
#   --stage init  只需前缀:决策步 = 浏览的最后一屏,问题与源帧留空(供 pilot_textmem.py --archive-only 建档案);
#   --stage fill  档案就绪后,**此时才**抽取要问的 (供应商, 属性)——采集期间任何记忆策略都不可能对准它;
#                 源帧 = OCR 转写里同时含该供应商名与两个以上报价字段的帧(即那封报价邮件被打开时的屏),不使用答案定位。
import argparse, glob, json, os, random, re

ap = argparse.ArgumentParser()
ap.add_argument("--prefix-dir", required=True); ap.add_argument("--facts", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--stage", choices=["init", "fill"], required=True)
ap.add_argument("--draw-seed", type=int, default=20260907)
args = ap.parse_args()

ATTRS = {"unit price": "unit price", "delivery charge": "delivery charge", "lead time": "lead time"}

def facts():
    out = {}
    for line in open(args.facts):
        m = re.search(r"BrowseRecords pair=(\d+) twin=(\d): facts=(\{.*\})", line.strip())
        if m: out[f"BrowseRecordsTask{int(m.group(1)):02d}{'AB'[int(m.group(2))]}"] = json.loads(m.group(3))
    return out

def shots_of(d):
    return sorted(glob.glob(os.path.join(d, "screenshots", "*.png")), key=lambda p: int(re.search(r"-(\d+)\.png$", p).group(1)))

F = facts(); rows = []
for d in sorted(glob.glob(os.path.join(args.prefix_dir, "*/"))):
    d = d.rstrip("/"); name = os.path.basename(d)
    if name not in F or not os.path.exists(os.path.join(d, "traj.json")): continue
    data = json.load(open(os.path.join(d, "traj.json")))
    if not data: continue
    traj = list(data.values())[0].get("traj") or []
    shots = shots_of(d)
    if len(traj) < 6 or len(shots) < 6: continue
    k = min(len(traj), len(shots))
    r = {"dir": d, "task": name, "family": "BrowseRecords", "pair": int(name[-3:-1]), "twin": 0 if name.endswith("A") else 1,
         "step": k, "n_steps": len(traj), "expected_kind": "text"}
    if args.stage == "init":
        r.update({"expected": "", "source_frames": [], "request_frames": []})
    else:
        rng = random.Random(args.draw_seed + hash(name) % 10000)
        sup = rng.choice(sorted(F[name])); attr = rng.choice(sorted(ATTRS))
        val = F[name][sup][attr]
        r["expected"] = f"{val:.2f}" if attr == "unit price" else str(val)
        r["target"] = {"supplier": sup, "attribute": attr}
        r["question"] = (f"Your manager just asked, without warning: what is the {attr} quoted by {sup}? "
                         f"Answer from what you remember of this session with Answer(content='<value>'); do not navigate.")
        arc_path = os.path.join(d, "archive_v1.json")
        src = []
        if os.path.exists(arc_path):
            arc = json.load(open(arc_path))
            for j, txt in sorted(arc["ocr"].items(), key=lambda kv: int(kv[0])):
                low = txt.lower()
                if sup.lower() in low and sum(a in low for a in ATTRS) >= 2: src.append(int(j))
        r["source_frames"] = src[-1:]           # 该邮件被打开的最后一帧
        r["request_frames"] = []
        r["valid"] = bool(src)
    rows.append(r)
with open(args.out, "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
n_ok = sum(r.get("valid", True) for r in rows)
print(f"stage={args.stage} prefixes={len(rows)} with_source={n_ok} -> {args.out}")
if args.stage == "fill":
    for r in rows[:6]:
        print(f"  {r['task']:22s} steps={r['n_steps']:2d} k={r['step']:2d} target={r['target']['supplier']}/{r['target']['attribute']} exp={r['expected']} src={r['source_frames']}")
