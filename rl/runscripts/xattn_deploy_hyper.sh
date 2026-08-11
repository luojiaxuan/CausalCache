#!/bin/bash
# note (luojiaxuan): selector 2.6 部署口径评测——重训(带 scorer 保存)+ argmax
# 查表评测,单种子单折一条链。容器内用法:
#   bash /data/xattn_deploy_hyper.sh <torch_seed> <fold> <cuda_id>
# 产物齐了自动跳过(deploy json 为完成标志),容器被杀后重跑即续。
# 首轮三种子(21:42/22:55/00:09 三份报告)没存权重,argmax 无从复现,故重训;
# lr=3e-4 两配置在三种子中从未进入最优,--configs 白名单里省掉(时间省 1/3)。
set -u
TS=$1; FOLD=$2; GPU=$3
D=/data/v5raw/deploy
mkdir -p "$D"
export PYTHONPATH=/data
LOG=$D/run_s${TS}_f${FOLD}.log
if [ -f "$D/deploy_s${TS}_f${FOLD}.json" ]; then
  echo "SKIP s${TS} f${FOLD} 已有产物" >> "$D/status.txt"
  exit 0
fi
CUDA_VISIBLE_DEVICES=$GPU python3 /data/train_selector_token_xattn.py \
  --labels /data/oracle/labels_all.jsonl \
  --token-dir /data/v5raw/tokens \
  --output "$D/report_s${TS}_f${FOLD}.json" \
  --torch-seed "$TS" --fold "$FOLD" \
  --configs "xattn@0.0001,xattn+recency@0.0001" \
  --save-dir "$D/scorers_s${TS}_f${FOLD}" \
  --max-ram-states 300 >> "$LOG" 2>&1 \
&& CUDA_VISIBLE_DEVICES=$GPU python3 /data/rl_selector_deploy_eval.py \
  --labels /data/oracle/labels_all.jsonl \
  --manifest /data/oracle/agentnet_screening_manifest_ubuntu_v1.jsonl \
  --domain-json /data/task_domain.json \
  --token-dir /data/v5raw/tokens \
  --scorers "$D/scorers_s${TS}_f${FOLD}"/*.pt \
  --fold "$FOLD" \
  --output "$D/deploy_s${TS}_f${FOLD}.json" >> "$LOG" 2>&1 \
&& echo "DONE s${TS} f${FOLD} $(date -u +%H:%M:%S)" >> "$D/status.txt" \
|| echo "FAIL s${TS} f${FOLD} $(date -u +%H:%M:%S)" >> "$D/status.txt"
