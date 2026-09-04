# note (luojiaxuan): 把 GUI-Owl 的成功轨迹(截图 + traj.json + result.txt)打成 ~1GB 的 tar 分片上传 HF,
# 供 Tilde 侧同口径重标使用。排除 *_polluted_* 与 *_seedbug*(污染/已知 bug 的批次)。
# 海量小文件必须先 tar 再传:裸传 18 万级 PNG 会撞 repo 提交与 API 限流(2026-08-21 教训)。
import glob, os, subprocess, sys, tarfile
SRC = "/data01/jaxan/mw"
OUT = "/data01/jaxan/pack_guiowl"; os.makedirs(OUT, exist_ok=True)
SHARD_BYTES = 1 << 30

dirs = []
for rp in sorted(glob.glob(f"{SRC}/traj_*/*/result.txt")):
    d = os.path.dirname(rp)
    if "_polluted_" in d or "seedbug" in d:
        continue
    try:
        if float(open(rp).read().split("score:")[1].split()[0]) <= 0:
            continue
    except (IndexError, ValueError):
        continue
    dirs.append(d)
print(f"trajectories: {len(dirs)}", flush=True)

shard = idx = 0
tar = None
manifest = open(f"{OUT}/manifest.txt", "w")
for d in dirs:
    if tar is None or os.path.getsize(f"{OUT}/shard_{shard:03d}.tar") > SHARD_BYTES:
        if tar is not None:
            tar.close(); shard += 1
        tar = tarfile.open(f"{OUT}/shard_{shard:03d}.tar", "w")
    keep = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs
            if f.endswith(".png") or f in ("traj.json", "result.txt")]
    for p in keep:
        tar.add(p, arcname=os.path.relpath(p, SRC))
    manifest.write(f"{os.path.relpath(d, SRC)}\t{len(keep)}\tshard_{shard:03d}.tar\n")
    idx += 1
    if idx % 50 == 0:
        print(f"{idx}/{len(dirs)} -> shard {shard}", flush=True)
if tar is not None:
    tar.close()
manifest.close()
print(f"shards={shard+1} total={sum(os.path.getsize(p) for p in glob.glob(OUT+'/shard_*.tar'))/2**30:.1f} GiB", flush=True)
