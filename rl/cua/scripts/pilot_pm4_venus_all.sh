#!/usr/bin/env bash
# note (luojiaxuan): PartMatch v4 的 Venus 四口径矩阵一次跑齐(整段文本 / 无文本 / 只留 action / action + 自己的一句描述),规格用绝对路径重建
# (相对路径的规格在别的 cwd 下找不到 traj.json,前两条链因此空跑)。起服务→跑→删。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot
python3 /data01/jaxan/pilot_build_specs.py --prefix-dir $P/prefix_venus_pm4 --seeds $P/task_seeds.txt --backend venus --out $P/specs_prefix_venus_pm4.jsonl > $P/specs_prefix_venus_pm4.log 2>&1
grep "valid_checkpoints" $P/specs_prefix_venus_pm4.log; head -c 200 $P/specs_prefix_venus_pm4.jsonl; echo
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41201 41211 41221 41231 41241 41271)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim -p 172.17.0.1:$PORT:8000 \
  vllm/vllm-omni:dev --model /data01/jaxan/models/UI-Venus-2-9b --served-model-name UI-Venus-2 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl B-pilot PartMatch v4 Venus 四口径矩阵;⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) venus up ($C GPU$G:$PORT)"
CONDS=text_only,rec2,ctrl_at_turn,irr_at_turn,src_at_turn,swap_at_turn,gold_text
run() { rm -f $P/eval_$1.jsonl; python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/specs_prefix_venus_pm4.jsonl --backend venus --base-url $E --model UI-Venus-2 --tag $1 --conds $CONDS --out $P/eval_$1.jsonl --workers 4 "${@:2}" 2>&1 | tee $P/eval_$1.txt; }
run pm4_venus_full --text-mode full
run pm4_venus_notext --no-text
run pm4_venus_action --text-mode action
run pm4_venus_action_fact --text-mode action_fact
rm_own "$C"; echo PILOT_PM4_VENUS_ALL_DONE
