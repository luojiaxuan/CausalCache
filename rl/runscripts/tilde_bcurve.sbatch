#!/bin/bash
#SBATCH --job-name=cc-bcurve
#SBATCH --partition=main
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# note (luojiaxuan): B 曲线重测 —— 在**新指标**(人标 gold + 步级正确率)下
# 重新量 B ∈ {1,2,4} 的 oracle 头寸。
#
# **为什么必须重测**:B=2 这个预算当初的依据是旧的 teacher-forced log-prob
# DiD 指标(oracle 相对 recent 的可提升空间 B=2 是 +0.102、B=4 只剩 +0.037),
# 而那套指标用了 target-recurrence 启发式筛候选、且与部署时的 argmax 脱钩,
# 已经不可信。同时 frozen_main_row_v1 显示 B2→B4 在真实任务成功率上还有
# **+4.6pt** —— 也就是说 B=2 是一个比 B=4 低 4.6pt 的工作点,审稿人完全可以问
# "为什么不在 B=4 上比?那里你的头寸小 2.8 倍"。这条曲线要么坐实 B=2,
# 要么换掉它,不能继续靠一份已被推翻的证据。
#
# **配对**:三个 B 必须打在同一批 state 上(--only-dp-ids),否则曲线的升降
# 会混进样本差异。B=2 那个点位直接复用 labels_all.jsonl 的全枚举结果,不重跑。
#
# **easy 态必须进样本**:easy 是按 B=0 定义的,加到 B=4 时多出来的历史图完全
# 可能把本来答对的题干扰错。净收益 = 大 B 修好的 winnable − 大 B 弄坏的 easy。
# 只测非 easy 会系统性偏向大 B。白名单 = 全部 2494 非 easy + 随机 800 easy。
#
# 断点续跑同 tilde_oracle_enumerate.sbatch:--requeue + 每分片独立 jsonl +
# 启动读回已完成 dp_id,最大重放一个 state。
#
# 用法:sbatch --export=ALL tilde_bcurve.sbatch
set -u
SHARDS=${SHARDS:-"0 1 2 3 4 5 6 7"}
SHARD_COUNT=${SHARD_COUNT:-10}     # 余下 8、9 两片在 hyper00 GPU6/7 上跑
LIMIT=${LIMIT:-4000}

ROOT=$HOME/causalcache
REPO=$ROOT/CausalCache
MODEL=$ROOT/models/GUI-Owl-1.5-8B-Instruct
IMAGES=$ROOT/agentnet
MANIFEST=$ROOT/agentnet/agentnet_screening_manifest_ubuntu_v1.jsonl
OUT=$ROOT/oracle
IDS=$OUT/bcurve_ids.txt
IMAGE=$ROOT/sglang-omni-dev.sqsh
mkdir -p "$OUT"
[ -f "$IDS" ] || { echo "缺 $IDS(配对白名单),先传上来"; exit 1; }

SNAP=$MODEL/.snapshot.json
[ -f "$SNAP" ] || cp "$REPO/code/configs/gui_owl_1_5_8b_snapshot.json" "$SNAP"
export SLURM_CPU_BIND=quiet

srun --container-image="$IMAGE" \
     --container-mounts="$ROOT:$ROOT" \
     --container-workdir="$REPO" \
     bash -lc '
set -u
i=0
for SH in '"$SHARDS"'; do
  (
    # note (luojiaxuan): 每张卡串行跑完 B=1 再跑 B=4。B=1 便宜(全枚举 n 个
    # 单帧子集,约 11 次前向/态),先出来就能先看到曲线左端;B=4 贵(剪枝后
    # 约 80 次/态)放后面。pylibs 必须排最左盖住镜像自带的 transformers。
    for spec in "1 full" "4 pruned"; do
      set -- $spec; B=$1; MODE=$2
      CUDA_VISIBLE_DEVICES='"$i"' PYTHONPATH='"$ROOT"'/pylibs:code:rl/code \
        python3 rl/code/scripts/rl_oracle_enumerate.py \
        --manifest '"$MANIFEST"' --image-root '"$IMAGES"' \
        --model-dir '"$MODEL"' --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
        --output '"$OUT"'/bcurve_b${B}_sh$SH.jsonl --limit-states '"$LIMIT"' \
        --max-candidates 30 --only-dp-ids '"$IDS"' \
        --budget $B --mode $MODE --top-k 8 \
        --shard-index $SH --shard-count '"$SHARD_COUNT"' \
        >> '"$OUT"'/bcurve_sh$SH.log 2>&1
    done
  ) &
  i=$((i+1))
done
wait
echo TILDE_BCURVE_DONE
'
