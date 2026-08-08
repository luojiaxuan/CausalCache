#!/bin/bash
#SBATCH --job-name=cc-oracle
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
# note (luojiaxuan): Tilde 上的枚举 oracle 作业(8 卡 = 8 个独立分片)。
#
# **为什么枚举适合 tilde 而 OSWorld RL 不适合**:枚举是纯 GPU 前向,
# 不需要 /dev/kvm、不需要 docker daemon —— 那两条正是当初判 tilde 跑不了
# OSWorld 的原因(计算节点两者皆无)。转到离线数据后这个阻塞消失了。
#
# **断点续跑**(全局规则:Slurm 作业提交前必须可重启):
#   * `--requeue` + `--open-mode=append`:被抢占/超时自动重排队,日志续写;
#   * 每个分片独立写自己的 jsonl,脚本启动时读回已完成的 dp_id 跳过 ——
#     **最大重放 = 一个 state**(约 25 秒),远小于作业时长;
#   * 输出落 $HOME(Tilde 上跨作业持久),不用节点本地盘 / 容器内路径 / /tmp。
#
# **分片必须与 hyper 对齐,否则重叠或漏测。** hyper 用 9 分片,正在跑
# 0,1,2(h00)与 5,6(h01),缺 3,4,7,8。这里用 **18 分片**:因为
# `行号 % 18 == k` 蕴含 `行号 % 9 == k mod 9`,所以下面这 8 片
#   {3,12} ∪ {4,13} ∪ {7,16} ∪ {8,17}
# 恰好覆盖 hyper 缺的 4 片(每片再对半切),8 卡刚好、零重叠零遗漏。
#
# 用法:sbatch --export=ALL tilde_oracle_enumerate.sbatch
set -u
SHARDS=${SHARDS:-"3 12 4 13 7 16 8 17"}
SHARD_COUNT=${SHARD_COUNT:-18}
LIMIT=${LIMIT:-900}

ROOT=$HOME/causalcache
REPO=$ROOT/CausalCache
MODEL=$ROOT/models/GUI-Owl-1.5-8B-Instruct
IMAGES=$ROOT/agentnet   # 注意:HF 解包落在 agentnet/ubuntu_images/extracted,manifest 相对路径含 ubuntu_images/,故 root 是 agentnet 而非再下一层
MANIFEST=$ROOT/agentnet/agentnet_screening_manifest_ubuntu_v1.jsonl
OUT=$ROOT/oracle
IMAGE=$ROOT/sglang-omni-dev.sqsh
mkdir -p "$OUT"

# 冻结守卫要模型目录带 .snapshot.json(HF 快照不带),幂等补齐;不绕过校验。
SNAP=$MODEL/.snapshot.json
[ -f "$SNAP" ] || cp "$REPO/code/configs/gui_owl_1_5_8b_snapshot.json" "$SNAP"

# 站点特性:默认 cpu-bind 带 verbose,不设就每 task 刷一行噪声
export SLURM_CPU_BIND=quiet

srun --container-image="$IMAGE" \
     --container-mounts="$ROOT:$ROOT" \
     --container-workdir="$REPO" \
     bash -lc '
set -u
i=0
for SH in '"$SHARDS"'; do
  # note (luojiaxuan): 冻结守卫钉死 transformers==5.6.0,而镜像自带的版本不同
  # (首次提交 8 个分片全部 ValueError: Transformers version drifted)。
  # $ROOT/pylibs 里装着 5.6.0,**必须排在最左**才能盖住镜像自带的 —— 顺序错了
  # 等于没装。这是 tilde 上此前 HGKV 作业已踩过并固化的做法。
  CUDA_VISIBLE_DEVICES=$i PYTHONPATH='"$ROOT"'/pylibs:code:rl/code \
    python3 rl/code/scripts/rl_oracle_enumerate.py \
    --manifest '"$MANIFEST"' --image-root '"$IMAGES"' \
    --model-dir '"$MODEL"' --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
    --output '"$OUT"'/sh$SH.jsonl --limit-states '"$LIMIT"' --max-candidates 30 \
    --budget 2 --mode pruned --top-k 8 --skip-easy \
    --shard-index $SH --shard-count '"$SHARD_COUNT"' \
    > '"$OUT"'/sh$SH.log 2>&1 &
  i=$((i+1))
done
wait
echo TILDE_ORACLE_DONE
'
