#!/usr/bin/env bash
# note (luojiaxuan): 32B HGKV offline DiD 门控(B=1,六 checkpoint),aux CUDA1=host GPU5。
set -uo pipefail
OUT=/bigdata/mw/runs/desktop-did-32b/devscore/hgkv-b1
mkdir -p "$OUT"
cd /data/CausalCache-mwhgkv
export PYTHONPATH=/data/CausalCache-mwhgkv/code
CK=/bigdata/mw/runs/desktop-did-32b/hgkv
for a in 1 2 3; do
  python3 code/scripts/score_sparse_history_arms.py \
    --repository-root . \
    --config code/configs/causalcache_desktop_did_hgkv_v4_32b.json \
    --model-dir /bigdata/models/GUI-Owl-1.5-32B-Instruct \
    --model-profile 32b \
    --dataset-root /bigdata/mw/desktop-did-corpus-v4/samples-b1.jsonl \
    --image-root /bigdata/mw/desktop-did-corpus-v4 \
    --checkpoint lora-step50=$CK/lora-step50.pt \
    --checkpoint lora-step100=$CK/lora-step100.pt \
    --checkpoint lora-step150=$CK/lora-step150.pt \
    --checkpoint lora-step200=$CK/lora-step200.pt \
    --checkpoint lora-step250=$CK/lora-step250.pt \
    --checkpoint lora-step300=$CK/lora-step300.pt \
    --score-cache $OUT/cache.jsonl \
    --heartbeat $OUT/heartbeat.json \
    --output $OUT/gate_report.json \
    --device cuda:1 >> $OUT/score.log 2>&1 && { touch $OUT/DEVSCORE_DONE; exit 0; }
  echo "attempt $a failed" >> $OUT/score.log
  sleep 20
done
exit 1
