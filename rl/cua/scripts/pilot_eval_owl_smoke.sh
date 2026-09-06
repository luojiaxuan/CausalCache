#!/usr/bin/env bash
# note (luojiaxuan): B-pilot 评测器的 GUI-Owl 端到端冒烟:用底座 GUI-Owl 自己的前缀里能建出的 checkpoint(目前只有 PartMatch 族到得了决策步)
# 跑一遍干预矩阵,验证构造器/判定/泄漏标志;等 Venus 前缀链拿到卡并登记后再取卡,避免与它抢同一张空卡。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
until grep -q PILOT_PREFIX_DONE /data01/jaxan/pilot_prefix.log 2>/dev/null; do sleep 60; done
until grep -q "venus up" /data01/jaxan/pilot_prefix_venus.log 2>/dev/null; do sleep 60; done
P=/data01/jaxan/rl_v2/pilot
for c in sglang-omni-jaxan-p12 sglang-omni-jaxan-p13; do docker logs $c 2>&1 | grep -a "pair="; done | sed "s/.*- //" | sort -u > $P/task_seeds.txt
python3 /data01/jaxan/pilot_build_specs.py --prefix-dir $P/prefix_base --seeds $P/task_seeds.txt --backend owl --out $P/specs_owl.jsonl --contact $P/contact_owl > $P/specs_owl.log 2>&1
n=$(wc -l < $P/specs_owl.jsonl); echo "$(date -u +%FT%TZ) owl specs=$n"; [ "$n" -gt 0 ] || { echo "NO_OWL_CHECKPOINTS"; exit 0; }
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41221 41231 41241 41271 41281)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim \
  -p 172.17.0.1:$PORT:8000 vllm/vllm-omni:dev --model /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct --served-model-name gui-owl \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl B-pilot 评测器冒烟(GUI-Owl 后端,PartMatch checkpoints);⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) owl up ($C GPU$G:$PORT)"
rm -f $P/eval_owl_smoke.jsonl
python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/specs_owl.jsonl --backend owl --base-url $E --model gui-owl --tag owl_smoke \
  --out $P/eval_owl_smoke.jsonl --workers 4 2>&1 | tee $P/eval_owl_smoke.txt
rm_own "$C"; echo PILOT_EVAL_OWL_SMOKE_DONE
