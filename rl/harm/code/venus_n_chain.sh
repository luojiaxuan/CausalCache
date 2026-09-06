#!/usr/bin/env bash
# note (luojiaxuan): Venus 的 N 扫描(400 态,保留文本轨迹),看 N≥4 的终止病理是否跨模型。
set -uo pipefail
source /data01/jaxan/cc_container_lib.sh
H=/data01/jaxan/harm; C=""
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 120; done
PORT=$(free_port); E=http://127.0.0.1:$PORT/v1
run_labeled --gpus "\"device=$G\"" --ipc=host --shm-size 16g --network host -v /data01/jaxan:/data01/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim \
  vllm/vllm-omni:dev --model /data01/jaxan/models/UI-Venus-2-9b --served-model-name UI-Venus-2 --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt '{"image":10}' --host 127.0.0.1 --port $PORT || { echo "RUN_FAIL"; exit 1; }
reg "$C" "$G" "sglang-omni-rl history-harm:Venus N 扫描(port $PORT);⚠ 在用勿删;收尾:出数即删"
for t in $(seq 1 60); do curl -s -m 5 $E/models | grep -q '"UI-Venus-2"' && break; sleep 20; done
curl -s -m 60 $E/chat/completions -H "Content-Type: application/json" -d '{"model":"UI-Venus-2","messages":[{"role":"user","content":"Say OK"}],"max_tokens":4}' | grep -q '"choices"' || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) venus up on GPU$G ($C)"
# note (luojiaxuan): 首发 HTTP 404——先用与脚本完全相同的 URL 构造探一次,404 就把响应体打出来退出。
PROBE=$(python3 - "$E" <<'PY'
import importlib.util, sys, json, urllib.request
spec = importlib.util.spec_from_file_location("vo", "/data01/jaxan/venus_oracle.py"); vo = importlib.util.module_from_spec(spec); spec.loader.exec_module(vo)
try:
    out = vo.post(sys.argv[1], {"model": "UI-Venus-2", "temperature": 0.0, "max_tokens": 8, "messages": [{"role": "user", "content": "Say OK"}]})
    print("PROBE_OK", (out["choices"][0]["message"]["content"] or "")[:20])
except urllib.error.HTTPError as e:
    print("PROBE_FAIL", e.code, e.read()[:200])
PY
); echo "$PROBE"; echo "$PROBE" | grep -q PROBE_OK || { rm_own "$C"; exit 1; }
python3 /data01/jaxan/decode_ctx_venus.py --base-url $E --tag venus_n --out $H/harm_vs_n_venus_text.jsonl --workers 8 > $H/harm_vs_n_venus_text.log 2>&1
echo "$(date -u +%FT%TZ) done: $(wc -l < $H/harm_vs_n_venus_text.jsonl)"; rm_own "$C"; echo VENUS_N_DONE
