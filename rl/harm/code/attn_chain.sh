#!/usr/bin/env bash
# note (luojiaxuan):注意力探针——HF 前向(非 vLLM),一张卡,40 态 × 4 规格,约 20–40 分钟。
set -uo pipefail
source /data01/jaxan/cc_container_lib.sh
H=/data01/jaxan/harm; C=""
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 120; done
run_labeled --gpus "\"device=$G\"" --ipc=host --shm-size 16g -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan --entrypoint bash vllm/vllm-omni:dev \
  -c "python3 /data01/jaxan/attn_probe.py --out $H/attn_probe_base.jsonl > $H/attn_probe_base.log 2>&1" || { echo "RUN_FAIL"; exit 1; }
reg "$C" "$G" "sglang-omni-rl history-harm:注意力探针(HF 前向);⚠ 在用勿删;收尾:出数即删"
echo "$(date -u +%FT%TZ) attn probe started on GPU$G ($C)"
docker wait "$C" >/dev/null; tail -3 $H/attn_probe_base.log | cut -c1-160; rm_own "$C"; echo ATTN_CHAIN_DONE
