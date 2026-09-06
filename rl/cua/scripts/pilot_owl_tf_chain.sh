#!/usr/bin/env bash
# note (luojiaxuan): 一行摘要 agent 的判别实验:等 Venus 文本保真度扫描释放卡 → 起 GUI-Owl → 沿 Venus 的 Mail 轨迹教师强制生成 GUI-Owl 自己的前缀
# → 同一批 checkpoint 上跑 GUI-Owl 部署协议的干预矩阵(保留哪几轮)→ 删。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot
until grep -q PILOT_VENUS_TEXTMODE_DONE /data01/jaxan/pilot_venus_textmode.log 2>/dev/null; do sleep 60; done
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41221 41231 41241 41271 41281)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim \
  -p 172.17.0.1:$PORT:8000 vllm/vllm-omni:dev --model /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct --served-model-name gui-owl \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl B-pilot GUI-Owl 教师强制前缀 + Mail 矩阵;⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) owl up ($C GPU$G:$PORT)"
rm -rf $P/prefix_owl_tf; mkdir -p $P/prefix_owl_tf
python3 /data01/jaxan/pilot_owl_tf.py --spec $P/specs_venus_mail.jsonl --base-url $E --out-root $P/prefix_owl_tf --workers 4 2>&1 | tail -3
rm -f $P/eval_owl_tf_mail.jsonl
python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/prefix_owl_tf/specs_owl_tf.jsonl --backend owl --base-url $E --model gui-owl --tag owl_tf_mail --out $P/eval_owl_tf_mail.jsonl --workers 4 2>&1 | tee $P/eval_owl_tf_mail.txt
python3 /data01/jaxan/pilot_gates.py $P/eval_owl_tf_mail.jsonl | tail -9
rm_own "$C"; echo PILOT_OWL_TF_DONE
