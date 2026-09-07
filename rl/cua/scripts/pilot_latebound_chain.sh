#!/usr/bin/env bash
# note (luojiaxuan): 外审 §3.1 的延后揭示半段:通用目标浏览(采集时不存在请求)→ 建档案 → 采集之后才抽问题 → 同预算下比各记忆策略。
# 装任务类需重启 p12/p13 的任务服务,故先等文本记忆基线链结束(它不用模拟器,但两条链共用 GPU 预算,顺序跑更稳)。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot; EMUS=$(cat /data01/jaxan/.pilot_emus)
until grep -q PILOT_TEXTMEM_DONE /data01/jaxan/pilot_textmem_chain.log 2>/dev/null; do sleep 60; done
bash /data01/jaxan/install_pilot_tasks.sh sglang-omni-jaxan-p12 sglang-omni-jaxan-p13 2>&1 | grep -c "OK:"
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41201 41211 41221 41231 41241 41271)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim -p 172.17.0.1:$PORT:8000 \
  vllm/vllm-omni:dev --model /data01/jaxan/models/UI-Venus-2-9b --served-model-name UI-Venus-2 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":16}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl 外审 §3.1 延后揭示(通用浏览采集 + 回忆探针,Venus);⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) venus up ($C GPU$G:$PORT)"
TASKS=$(python3 -c "print(','.join(f'BrowseRecordsTask{k:02d}{t}' for k in range(1,9) for t in 'AB'))")
AW=$(echo "$EMUS" | tr ',' '\n' | sed 's#^#http://127.0.0.1:#' | paste -sd,); NC=$(echo "$EMUS" | tr ',' '\n' | wc -l)
rm -rf $P/prefix_browse; mkdir -p $P/prefix_browse; cd /data01/jaxan/mw/MobileWorld
CC_VENUS_HIST=recent:2 PYTHONPATH=/data01/jaxan/pyshim timeout 7200 uv run mw eval --agent_type ui_venus2 --task "$TASKS" --max_round 30 \
  --model_name UI-Venus-2 --llm_base_url $E --api_key EMPTY --step_wait_time 3 --max-concurrency $NC --aw-host "$AW" \
  --log_file_root $P/prefix_browse > $P/prefix_browse.log 2>&1
echo "$(date -u +%FT%TZ) DONE collect dirs=$(ls -d $P/prefix_browse/*/ 2>/dev/null | wc -l)"
for c in sglang-omni-jaxan-p12 sglang-omni-jaxan-p13; do docker logs --since 60m $c 2>&1 | grep -a "BrowseRecords pair="; done | sed "s/.*- //" | sort -u > $P/facts_browse.txt
wc -l < $P/facts_browse.txt
python3 /data01/jaxan/pilot_browse_specs.py --prefix-dir $P/prefix_browse --facts $P/facts_browse.txt --out $P/specs_browse_init.jsonl --stage init
python3 /data01/jaxan/pilot_textmem.py --spec $P/specs_browse_init.jsonl --base-url $E --tag archive --out /dev/null --archive-only --workers 4 2>&1 | tail -2
python3 /data01/jaxan/pilot_browse_specs.py --prefix-dir $P/prefix_browse --facts $P/facts_browse.txt --out $P/specs_browse.jsonl --stage fill
rm -f $P/eval_latebound.jsonl
python3 /data01/jaxan/pilot_textmem.py --spec $P/specs_browse.jsonl --base-url $E --tag latebound --probe --out $P/eval_latebound.jsonl --workers 4 2>&1 | tee $P/eval_latebound.txt
rm_own "$C"; echo PILOT_LATEBOUND_DONE
