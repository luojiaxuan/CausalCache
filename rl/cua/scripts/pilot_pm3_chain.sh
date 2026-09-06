#!/usr/bin/env bash
# note (luojiaxuan): PartMatch 第三版(Documents/Candidates 列表视图)全链:等 QuoteRecall 重采链结束释放模拟器 → 装任务类 →
# Venus 采 16 题 → 规格 → 矩阵(原生文本 + 无文本)→ 删;GUI-Owl 同流程(只有原生文本)。每段一台执行器,跑完即删。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot; EMUS=$(cat /data01/jaxan/.pilot_emus)
until grep -q PILOT_QR2_DONE /data01/jaxan/pilot_qr_recollect.log 2>/dev/null; do sleep 60; done
bash /data01/jaxan/install_pilot_tasks.sh sglang-omni-jaxan-p12 sglang-omni-jaxan-p13 2>&1 | grep -c "OK:"
TASKS=$(python3 -c "print(','.join(f'PartMatchTask{k:02d}{t}' for k in range(1,9) for t in 'AB'))")
AW=$(echo "$EMUS" | tr ',' '\n' | sed 's#^#http://127.0.0.1:#' | paste -sd,); NC=$(echo "$EMUS" | tr ',' '\n' | wc -l)
arm() {  # $1 agent_type $2 model path $3 served $4 backend $5 out prefix dir $6 extra env $7 protos
  local G="" PORT C E
  until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
  PORT=$(pick_port 41201 41211 41221 41231 41241 41271)
  C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
    -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim -p 172.17.0.1:$PORT:8000 \
    vllm/vllm-omni:dev --model "$2" --served-model-name "$3" --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; return 1; }
  reg "$C" "$G" "sglang-omni-rl B-pilot PartMatch v3 采集+矩阵($3);⚠ 在用勿删;收尾:出数即删"
  E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
  curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL $3"; rm_own "$C"; return 1; }
  echo "$(date -u +%FT%TZ) $3 up ($C GPU$G:$PORT)"
  rm -rf "$5"; mkdir -p "$5"; cd /data01/jaxan/mw/MobileWorld
  env $6 PYTHONPATH=/data01/jaxan/pyshim timeout 7200 uv run mw eval --agent_type "$1" --task "$TASKS" --max_round 40 --model_name "$3" \
    --llm_base_url $E --api_key EMPTY --step_wait_time 3 --max-concurrency $NC --aw-host "$AW" --log_file_root "$5" > "$5.log" 2>&1
  echo "$(date -u +%FT%TZ) DONE collect $3 dirs=$(ls -d $5/*/ 2>/dev/null | wc -l) succ=$(grep -l "^score: 1" $5/*/result.txt 2>/dev/null | wc -l)"
  local tag=$(basename "$5"); python3 /data01/jaxan/pilot_build_specs.py --prefix-dir "$5" --seeds $P/task_seeds.txt --backend $4 --out $P/specs_$tag.jsonl --contact $P/contact_$tag > $P/specs_$tag.log 2>&1
  grep -v "^    " $P/specs_$tag.log | head -3
  for proto in $7; do extra=$([ "$proto" = notext ] && echo --no-text || true); rm -f $P/eval_${tag}_$proto.jsonl
    python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/specs_$tag.jsonl --backend $4 --base-url $E --model "$3" --tag ${tag}_$proto $extra --out $P/eval_${tag}_$proto.jsonl --workers 4 2>&1 | tee $P/eval_${tag}_$proto.txt; done
  rm_own "$C"
}
arm ui_venus2 /data01/jaxan/models/UI-Venus-2-9b UI-Venus-2 venus $P/prefix_venus_pm3 CC_VENUS_HIST=recent:2 "text notext"
arm gui_owl_1_5 /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct gui-owl owl $P/prefix_base_pm3 "CC_HISTORY_N=3 CC_FRAME_POLICY=recent" "text"
echo PILOT_PM3_DONE
