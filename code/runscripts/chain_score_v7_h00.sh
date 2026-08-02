#!/bin/bash
# poolrank 训完 → 按预注册协议打门控(corpus-v4 的 94 组 dev + v4 config)
set -u
R=/data/runs/desktop-did-v7
for i in $(seq 1 300); do [ -f "$R/poolrank/EXIT_0" ] && break; sleep 30; done
[ -f "$R/poolrank/EXIT_0" ] || { echo POOLRANK_NEVER_FINISHED; exit 1; }
export MODEL_DIR=/data/artifacts/models/GUI-Owl-1.5-8B-Instruct
export DEV_SAMPLES=/data/desktop-did-corpus-v4/samples-b1.jsonl
export IMAGE_ROOT=/data/desktop-did-corpus-v4
export CONFIG=code/configs/causalcache_desktop_did_hgkv_v4.json
bash /data/CausalCache/code/runscripts/devscore_v6.sh poolrank 4 "$R" /data/CausalCache
