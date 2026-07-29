#!/usr/bin/env bash
# 用法: osworld_serve_loop_h00.sh <device_idx> <port> <log> [adapter] [selector]
# note (luojiaxuan): hyper00 版 OSWorld policy 重启循环(路径按 canonical 容器布局)。
set -uo pipefail
DEV=$1; PORT=$2; LOG=$3; ADAPTER=${4:-0}; SELECTOR=${5:-0}
REPO=/data/CausalCache
EXTRA=()
if [ "$ADAPTER" = "1" ]; then
  EXTRA+=(--adapter-checkpoint /data/runs/desktop-did-v4/hgkv/lora-step300.pt
          --adapter-checkpoint-sha256 572092c218a97d1aa88b6845086a2ad2cec77c1a6d6fcd796f2e125ed62d67a9)
fi
if [ "$SELECTOR" = "1" ]; then
  EXTRA+=(--selector-bundle /data/runs/selector-v4/arm-twotower/marginal_scorer.pt
          --selector-arch two_tower --selector-beam 3)
fi
BUDGET=4
[ "$ADAPTER" = "0" ] && [ "$SELECTOR" = "0" ] && BUDGET=0
cd "$REPO"
while true; do
  CUDA_VISIBLE_DEVICES=$DEV PYTHONPATH=code python3 code/scripts/serve_osworld_official_policy.py \
    --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
    --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
    --device cuda:0 --port "$PORT" --visual-tokens 2560 \
    --memory-budget "$BUDGET" "${EXTRA[@]}" >> "$LOG" 2>&1
  code=$?
  echo "{\"event\":\"POLICY_REPLICA_EXIT\",\"port\":$PORT,\"exit_code\":$code,\"at\":\"$(date -u +%FT%TZ)\"}" >> "$LOG"
  sleep 10
done
