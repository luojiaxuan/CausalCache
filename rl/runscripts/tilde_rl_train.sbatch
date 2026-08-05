#!/bin/bash
#SBATCH --job-name=ccrl-train
#SBATCH --partition=main
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out
# note (luojiaxuan): CausalCache-RL 的**训练步**上 tilde(2026-08-05 实探:
# 计算节点无 /dev/kvm 且 docker socket 拒绝,OSWorld VM 上不来——rollout 留 hyper,
# 跨站经 HF 私有仓库交换:轨迹+截图上行(~0.5GB/迭代),权重下行(~3MB)。
#
# 成本账(B=2,训练 rollout cap 30 步):
#   rollout(hyper,32 VM):16 任务 × G=8 ≈ 30 min
#   训练步:128 ep × ~25 步 × 3 前向(logprob+KL两遍)≈ 10 GPU·h
#     → 8×H100 数据并行 ≈ 75 min;步子采样(每 ep 抽 12 步)≈ 40 min
#   流水线重叠后 ~1h/迭代;60 迭代判停线 ≈ 2.5–3 天,100 迭代 ≈ 4–5 天。
# B=2 相对 B=4:prompt 短 ~25%、PL 采样 2 轮,rollout 与训练各省 20–30%。
#
# 断点续跑:trainer 每迭代原子写 selector_bundle.pt/adapter.pt;--requeue 后
# 从 HF 最新迭代目录继续。用法:
#   sbatch --export=ALL,ITER=7,HF_REPO=gavinlaw/causalcache-rl-osworld tilde_rl_train.sbatch
set -u
ITER=${ITER:?}
HF_REPO=${HF_REPO:-gavinlaw/causalcache-rl-osworld}
ROOT=$HOME/causalcache
REPO=$ROOT/CausalCache
OUT=$ROOT/rl/iter-$ITER
mkdir -p "$OUT"
export SLURM_CPU_BIND=quiet HF_HOME=$ROOT/hf
V=$ROOT/venv/bin/python
# 拉本迭代轨迹(hyper 侧 collect 后推 HF:groups.jsonl + attempts 截图包)
$V - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$HF_REPO", repo_type="dataset",
                  allow_patterns=["iter-$ITER/rollout/*"],
                  local_dir="$ROOT/rl")
PY
srun --container-image=$ROOT/sglang-omni-dev.sqsh \
     --container-mounts=$ROOT:$ROOT --container-workdir=$REPO \
     bash -lc "
       export PYTHONPATH=$ROOT/pylibs:$REPO/code:$REPO/rl/code
       python3 rl/code/scripts/train_causalcache_rl_grpo.py \
         --groups $ROOT/rl/iter-$ITER/rollout/groups.jsonl \
         --model-dir $ROOT/models/GUI-Owl-1.5-8B-Instruct \
         --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
         --selector-bundle $ROOT/rl/iter-$((ITER-1))/selector_bundle.pt \
         --resume-adapter $ROOT/rl/iter-$((ITER-1))/adapter.pt \
         --adapter-config code/configs/causalcache_desktop_did_hgkv_v7_poolrank.json \
         --output-root $OUT
     "
CODE=$?
[ $CODE -eq 0 ] && $V - <<PY
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path="$OUT", path_in_repo="iter-$ITER",
                  repo_id="$HF_REPO", repo_type="dataset")
print("weights pushed iter-$ITER")
PY
exit $CODE
