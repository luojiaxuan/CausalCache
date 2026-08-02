#!/bin/bash
set -u
R=/bigdata/mw/runs/desktop-did-v7
for i in $(seq 1 300); do [ -f "$R/didbase/EXIT_0" ] && break; sleep 30; done
[ -f "$R/didbase/EXIT_0" ] || { echo DIDBASE_NEVER_FINISHED; exit 1; }
export MODEL_DIR=/bigdata/models/GUI-Owl-1.5-8B-Instruct
export DEV_SAMPLES=/bigdata/mw/desktop-did-corpus-v4/samples-b1.jsonl
export IMAGE_ROOT=/bigdata/mw/desktop-did-corpus-v4
export CONFIG=code/configs/causalcache_desktop_did_hgkv_v4.json
bash /bigdata/osworld/CausalCache/code/runscripts/devscore_v6.sh didbase 0 "$R" /bigdata/osworld/CausalCache
