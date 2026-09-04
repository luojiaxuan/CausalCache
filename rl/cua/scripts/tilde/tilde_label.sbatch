#!/usr/bin/env bash
#SBATCH --job-name=jaxan-guiowl-label
#SBATCH --partition=main
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=%x-%j.out
# note (luojiaxuan): GUI-Owl 同口径标注。guest 账号 8 卡是**账号总额**且与其他租户共享,故只要 4 卡留出余量。
# 作业可被抢占/requeue:标注器按 (dir,step) 断点续跑,重跑即从盘上接着做,不重复解码。
set -uo pipefail
R=$HOME/jaxan/runs/guiowl_label
NG=4
cd $R
# note (luojiaxuan): 本机(Tilde,家目录在 VIRTIOFS 上)起 vLLM 的两处坑:
#  1) FlashInfer 的采样 kernel 在启动 dummy run 阶段就崩(job 278983/279062,四台确定性同错),
#     VLLM_USE_FLASHINFER_SAMPLER=0 走 PyTorch 原生采样绕开;
#  2) 编译缓存:四台共用一套目录有竞态风险,各给一套,并用 --enforce-eager 跳过 inductor 编译。
for i in $(seq 0 $((NG-1))); do
  mkdir -p $R/cache/$i/{triton,vllm,inductor}
  CUDA_VISIBLE_DEVICES=$i \
  TRITON_CACHE_DIR=$R/cache/$i/triton VLLM_CACHE_ROOT=$R/cache/$i/vllm TORCHINDUCTOR_CACHE_DIR=$R/cache/$i/inductor \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  $R/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model $R/GUI-Owl-1.5-8B-Instruct --served-model-name gui-owl --enforce-eager \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' \
    --port $((8001+i)) --host 127.0.0.1 > $R/logs/server_$i.log 2>&1 &
  sleep 10
done
# note (luojiaxuan): server 起不来时不能无限等——否则作业挂满 8 小时才被超时杀掉,白占卡。
for i in $(seq 0 $((NG-1))); do
  for t in $(seq 1 60); do
    curl -s -m 5 http://127.0.0.1:$((8001+i))/v1/models > /dev/null && break
    kill -0 %$((i+1)) 2>/dev/null || { echo "server $i died; see logs/server_$i.log"; exit 1; }
    sleep 20
  done
  curl -s -m 5 http://127.0.0.1:$((8001+i))/v1/models > /dev/null || { echo "server $i timeout"; exit 1; }
  echo "server $i up $(date -u +%FT%TZ)"
done
for i in $(seq 0 $((NG-1))); do
  $R/venv/bin/python $R/guiowl_oracle.py --base-url http://127.0.0.1:$((8001+i))/v1 --model gui-owl \
    --roots $R/data/traj_* --prompts $R/prompts.py --out $R/labels_base_shard$i.jsonl \
    --per-traj 12 --limit 600 --workers 12 --shard $i --nshards $NG > $R/logs/label_$i.log 2>&1 &
done
wait
wc -l $R/labels_base_shard*.jsonl
echo GUIOWL_LABEL_DONE
