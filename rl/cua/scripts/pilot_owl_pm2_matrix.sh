#!/usr/bin/env bash
# note (luojiaxuan): PartMatch(Pictures 布局)GUI-Owl 异质性臂:等 GUI-Owl 前缀 16 题齐 → 建规格 → 起 GUI-Owl 执行器跑矩阵(部署协议保留文本)→ 删。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot
until grep -q PILOT_PARTMATCH_OWL_DONE /data01/jaxan/pilot_partmatch_owl.log 2>/dev/null; do sleep 60; done
python3 /data01/jaxan/pilot_build_specs.py --prefix-dir $P/prefix_base_pm2 --seeds $P/task_seeds.txt --backend owl --out $P/specs_owl_pm2.jsonl --contact $P/contact_owl_pm2 > $P/specs_owl_pm2.log 2>&1
n=$(wc -l < $P/specs_owl_pm2.jsonl); echo "$(date -u +%FT%TZ) owl pm2 specs=$n"; grep -v "^    " $P/specs_owl_pm2.log | head -3
[ "$n" -gt 0 ] || { echo "NO_OWL_PM2_CHECKPOINTS"; exit 0; }
SPEC=$P/specs_owl_pm2.jsonl TAG=owl_pm2 bash /data01/jaxan/pilot_eval_owl_run.sh
echo PILOT_OWL_PM2_MATRIX_DONE
