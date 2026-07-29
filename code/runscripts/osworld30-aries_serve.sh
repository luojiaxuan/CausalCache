#!/usr/bin/env bash
set -uo pipefail
REPO=/data/osworld30/CausalCache
cd "$REPO"
while true; do
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=code python3 code/scripts/serve_osworld_official_policy.py \
    --model-dir /data/osworld30/GUI-Owl-1.5-8B-Instruct \
    --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
    --device cuda:0 --port 19380 --visual-tokens 2560 \
    --memory-budget 4 \
    --selector-bundle /data/osworld30/ship/runs/selector-v4/arm-twotower/marginal_scorer.pt \
    --selector-arch two_tower --selector-beam 3 >> /data/osworld30/s19380.log 2>&1
  echo "REPLICA_EXIT_$? $(date -u +%FT%TZ)" >> /data/osworld30/s19380.log
  sleep 10
done
