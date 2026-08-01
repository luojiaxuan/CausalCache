#!/usr/bin/env python3
# note (luojiaxuan): 构造"只补 aries 污染部分"的 meta。
# sel@30 由 aries(120)+ h100(251)合并而成,合并顺序 h100 优先,所以 10 个重叠任务
# 已用 h100 的结果;真正只有 aries 数据的是 361-251=110 个。只补这 110 个。
# 输出格式与官方 test_nogdrive.json 一致:{domain: [uuid, ...]}。
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
H = Path("/Users/luojiaxuan/Documents/CausalCache/.claude/worktrees/"
         "laughing-hodgkin-53a513/data/results/osworld30_horizon_v1")

aries = json.loads((H / "osw30_sel_aries.json").read_text())
h100 = json.loads((H / "osw30_sel_h100.json").read_text())

need = sorted(set(aries) - set(h100))
assert len(need) == 110, len(need)

meta: dict[str, list[str]] = {}
for t in need:
    meta.setdefault(aries[t].get("domain", "unknown"), []).append(t)
for d in meta:
    meta[d].sort()

out = ROOT / "osw_aries_repair_meta.json"
out.write_text(json.dumps(meta, indent=1))
print(f"任务数 {sum(len(v) for v in meta.values())}")
for d, v in sorted(meta.items()):
    print(f"  {d}: {len(v)}")
print(f"写入 {out}")
