#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 8 ]]; then
  echo "usage: $0 REPO_ROOT RUN_ROOT DATA_ROOT MODEL_DIR EPISODE_LIMIT GPU_ID LOGICAL_SHARD TOTAL_SHARDS" >&2
  exit 2
fi

REPO_ROOT="$1"
RUN_ROOT="$2"
DATA_ROOT="$3"
MODEL_DIR="$4"
EPISODE_LIMIT="$5"
GPU_ID="$6"
LOGICAL_SHARD="$7"
TOTAL_SHARDS="$8"
CONFIG="$REPO_ROOT/code/configs/findingdory_object_attribute_oracle_gap_v1.json"
MODEL_MANIFEST="$REPO_ROOT/code/configs/qwen2_5_vl_7b_snapshot.json"
export PYTHONPATH="$REPO_ROOT/code"

if (( LOGICAL_SHARD < 0 || LOGICAL_SHARD >= TOTAL_SHARDS )); then
  echo "logical shard must be in [0, total shards)" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs" "$DATA_ROOT" "$MODEL_DIR"
date --iso-8601=seconds > "$RUN_ROOT/started_at.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/git_commit.txt"
printf '%q ' "$0" "$@" > "$RUN_ROOT/argv.txt"
printf '\n' >> "$RUN_ROOT/argv.txt"
nvidia-smi -i "$GPU_ID" -q > "$RUN_ROOT/nvidia-smi-launch.txt"
python3 - <<'PY' > "$RUN_ROOT/runtime_versions.json"
import json
import platform
import torch
import transformers
print(json.dumps({
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
}, indent=2, sort_keys=True))
PY

if [[ ! -f "$MODEL_DIR/.snapshot.json" ]]; then
  python3 "$REPO_ROOT/code/scripts/download_hf_snapshot.py" \
    --manifest "$MODEL_MANIFEST" \
    --output-dir "$MODEL_DIR" \
    > "$RUN_ROOT/logs/model-download.log" 2>&1
fi

python3 "$REPO_ROOT/code/scripts/run_findingdory_oracle_gap.py" prepare-data \
  --config "$CONFIG" --data-dir "$DATA_ROOT" --output-dir "$RUN_ROOT" \
  --episode-limit "$EPISODE_LIMIT" \
  > "$RUN_ROOT/logs/prepare-data.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 "$REPO_ROOT/code/scripts/run_findingdory_oracle_gap.py" summarize \
  --config "$CONFIG" --data-dir "$DATA_ROOT" --model-dir "$MODEL_DIR" \
  --input-dir "$RUN_ROOT" --output-dir "$RUN_ROOT" --episode-limit "$EPISODE_LIMIT" \
  --device cuda:0 --shard-index "$LOGICAL_SHARD" --num-shards "$TOTAL_SHARDS" \
  > "$RUN_ROOT/logs/summarize.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 "$REPO_ROOT/code/scripts/run_findingdory_oracle_gap.py" evaluate \
  --config "$CONFIG" --data-dir "$DATA_ROOT" --model-dir "$MODEL_DIR" \
  --input-dir "$RUN_ROOT" --output-dir "$RUN_ROOT" --episode-limit "$EPISODE_LIMIT" \
  --device cuda:0 --shard-index "$LOGICAL_SHARD" --num-shards "$TOTAL_SHARDS" \
  > "$RUN_ROOT/logs/evaluate.log" 2>&1

date --iso-8601=seconds > "$RUN_ROOT/completed_at.txt"
touch "$RUN_ROOT/WORKER_COMPLETE"
