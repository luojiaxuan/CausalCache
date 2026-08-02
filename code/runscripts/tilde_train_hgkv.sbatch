#!/bin/bash
#SBATCH --job-name=cc-hgkv
#SBATCH --partition=main
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# note (luojiaxuan): Tilde(Slurm,main 分区 36×8 H100)上的 HGKV 训练作业。
#
# **断点续跑是这份脚本存在的理由**(全局规则:Slurm 作业提交前必须可重启)。
#   * `--requeue` + `--open-mode=append`:被抢占/超时后作业自动重排队,日志续写
#     而不是覆盖 —— 覆盖了就看不出这次是第几次重启。
#   * trainer 每 50 步写一份 `resume_state.pt`(global_step + optimizer 动量 +
#     adapter + config_sha),`--auto-resume` 默认开,重启从那里接。
#     **最大重放 = 50 步**,这是这份脚本对"可接受重放"的定义。
#   * 输出根目录必须落在**持久**存储:$HOME 在 Tilde 上跨作业保留;
#     绝不用节点本地盘、容器内路径或 /tmp(Tilde 的 /tmp 还是会话隔离的)。
#
# 站点特性两条:
#   * `SLURM_CPU_BIND=quiet` —— 站点默认带 verbose 修饰符,不设就每个 task 打一行
#     `cpu-bind=MASK ...`;`srun --quiet` 收不住它。
#   * 容器走 pyxis 的 `--container-image`,直接吃 enroot 导出的 .sqsh。
#     私有镜像 jaxanluo/sglang-omni:dev 匿名拉不到,用 hongccc/sglang-omni:dev。
#
# 用法:sbatch --export=ALL,VARIANT=poolrank_seed2,SEED=7 tilde_train_hgkv.sbatch
set -u
VARIANT=${VARIANT:-poolrank_seed2}
CONFIG=${CONFIG:-code/configs/causalcache_desktop_did_hgkv_v7_poolrank.json}
ROOT=$HOME/causalcache
REPO=$ROOT/CausalCache
CORPUS=$ROOT/v7pool/corpus
MODEL=$ROOT/models/GUI-Owl-1.5-8B-Instruct
OUT=$ROOT/runs/$VARIANT
IMAGE=$ROOT/sglang-omni-dev.sqsh

mkdir -p "$OUT"
# 持久性自检:输出根目录必须可写,且**不是** /tmp 或节点本地盘。
case "$OUT" in
  /tmp/*|/scratch/*|/dev/shm/*) echo "FATAL 输出目录不持久: $OUT"; exit 2 ;;
esac
touch "$OUT/.writable" || { echo "FATAL 输出目录不可写: $OUT"; exit 2; }
rm -f "$OUT/.writable"

echo "=== restart $(date -u +%FT%TZ) job=$SLURM_JOB_ID restarts=${SLURM_RESTART_COUNT:-0} node=$SLURMD_NODENAME ==="
[ -f "$OUT/resume_state.pt" ] && echo "发现 resume_state.pt,将从中续跑" || echo "无 resume_state.pt,从头开始"

export SLURM_CPU_BIND=quiet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun --container-image="$IMAGE" \
     --container-mounts="$ROOT:$ROOT" \
     --container-workdir="$REPO" \
     bash -lc "
       export PYTHONPATH=$REPO/code
       export HF_HOME=$ROOT/hf
       torchrun --standalone --nproc_per_node=8 code/scripts/train_success_sft_lora.py \
         --config $CONFIG \
         --dataset-root $CORPUS \
         --model-dir $MODEL \
         --output-root $OUT \
         --frozen-score-cache $OUT/frozen-cache \
         --encode-cache-scope group \
         --auto-resume \
         --repository-root .
     "
CODE=$?
echo \"=== exit $CODE $(date -u +%FT%TZ) ===\"
[ $CODE -eq 0 ] && touch "$OUT/EXIT_0"
exit $CODE
