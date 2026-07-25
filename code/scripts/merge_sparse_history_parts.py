import collections
import glob
import hashlib
import json
import os

SRC = "/bigdata/artifacts/sft/sparse-v5"
DST = "/bigdata/artifacts/sft/sparse-v5-final"
os.makedirs(DST, exist_ok=True)

parts = sorted(
    glob.glob(SRC + "/part*/samples.jsonl"),
    key=lambda p: int(p.split("/part")[1].split("/")[0]),
)
merged = 0
K = collections.Counter()
acts = collections.Counter()
rej = collections.Counter()
splits = collections.Counter()
trajs = 0
groups = 0

with open(DST + "/samples.jsonl", "w", encoding="utf-8") as out:
    for sp in parts:
        pdir = sp.rsplit("/", 1)[0]
        pname = os.path.basename(pdir)
        link = os.path.join(DST, pname)
        if not os.path.islink(link):
            os.symlink(pdir, link)
        for line in open(sp, encoding="utf-8"):
            s = json.loads(line)
            # note (luojiaxuan): 字段契约声明 selected_images / current_image 是
            # dataset_root 相对路径。合并后 dataset_root 就是本目录,所以这三处都要
            # 加 partN/ 前缀。v4 只改了 messages,另外两个字段在合并集上其实是错的,
            # 只是训练侧不读它们才没暴露 —— 审计要复核这些路径时就会踩到。
            s["current_image"] = pname + "/" + s["current_image"]
            s["selected_images"] = [pname + "/" + p for p in s["selected_images"]]
            for m in s["messages"]:
                for c in m["content"]:
                    if c.get("type") == "image":
                        c["path"] = pname + "/" + c["path"]
            out.write(json.dumps(s, ensure_ascii=False) + "\n")
            merged += 1
        mf = json.load(open(pdir + "/manifest.json"))
        trajs += mf.get("trajectories", 0)
        groups += mf.get("decision_groups", 0)
        for k, v in (mf.get("K_mix") or {}).items():
            K[k] += v
        for k, v in (mf.get("target_action_mix") or {}).items():
            acts[k] += v
        for k, v in (mf.get("rejected_reasons") or {}).items():
            rej[k] += v
        for k, v in (mf.get("groups_by_split") or {}).items():
            splits[k] += v

manifest = {
    "schema_version": "causalcache.sparse_history_dataset.v2",
    "source_parts": len(parts),
    "trajectories": trajs,
    "decision_groups": groups,
    "samples": merged,
    "K_mix": dict(sorted(K.items())),
    "target_action_mix": dict(sorted(acts.items())),
    "groups_by_split": dict(splits),
    "rejected_reasons": dict(sorted(rej.items())),
    "pool_repo": "cua-lite/GUIOdyssey",
    "pool_revision": "ea08072b30e523fb4492e4f4597505879ffcd63b",
    "model_repo": "mPLUG/GUI-Owl-1.5-8B-Instruct",
    "model_revision": "06d5faecff74840bab2be2425e9c42667a5d04fc",
}
json.dump(manifest, open(DST + "/manifest.json", "w"), ensure_ascii=False, indent=2)
digest = hashlib.sha256(open(DST + "/samples.jsonl", "rb").read()).hexdigest()
print("merged samples:", merged, "| groups:", groups, "| parts:", len(parts))
print("K_mix:", dict(sorted(K.items())), "| splits:", dict(splits))
print("merged samples.jsonl sha256:", digest)
