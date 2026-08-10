#!/bin/bash
#SBATCH --job-name=cc-bc2
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
# note (luojiaxuan): B 曲线扩样(C1):对未测过的 2591 个态补 B=1 与 B=2。
#
# **为什么只补 B1/B2 不补 B4**:扩样要回答的唯一悬而未决的问题是
# "recent-2 − recent-1 = +1.3pp(p=0.275)到底是真平还是样本不够" ——
# 这是 B 曲线非单调结论的软肋;B2→B4 的 +6.3pp 已经 p=0.000,不缺样本。
# B4 每态约 171 次前向且单价贵 1.37 倍,收益配不上成本。
#
# 分片按 manifest 行号 % 8,与 hyper 的产出按 dp_id 白名单天然无重叠
# (白名单本身就是差集)。逐态续跑 + requeue,排队/抢占无损。
# 合并归约时注意:easy 层是两次独立抽样的并集,按 dp_id 去重后分层重加权。
set -u
SHARD_COUNT=8
ROOT=$HOME/causalcache
REPO=$ROOT/CausalCache
MODEL=$ROOT/models/GUI-Owl-1.5-8B-Instruct
IMAGES=$ROOT/agentnet
MANIFEST=$ROOT/agentnet/agentnet_screening_manifest_ubuntu_v1.jsonl
OUT=$ROOT/oracle
IMAGE=$ROOT/sglang-omni-dev.sqsh
for f in "$OUT/bc2_ids_all.txt" "$OUT/bc2_ids_hard.txt" "$OUT/bc2_ids_easy.txt"; do
  [ -f "$f" ] || { echo "缺 $f"; exit 1; }
done
SNAP=$MODEL/.snapshot.json
[ -f "$SNAP" ] || cp "$REPO/code/configs/gui_owl_1_5_8b_snapshot.json" "$SNAP"
export SLURM_CPU_BIND=quiet

srun --container-image="$IMAGE" \
     --container-mounts="$ROOT:$ROOT" \
     --container-workdir="$REPO" \
     bash -lc '
set -u
i=0
while [ $i -lt 8 ]; do
  (
    SH=$i
    run () {
      B=$1; MODE=$2; IDS=$3; EXTRA=${4:-}
      for attempt in 1 2 3 4 5; do
        CUDA_VISIBLE_DEVICES=$SH PYTHONPATH='"$ROOT"'/pylibs:code:rl/code \
          python3 rl/code/scripts/rl_oracle_enumerate.py \
          --manifest '"$MANIFEST"' --image-root '"$IMAGES"' \
          --model-dir '"$MODEL"' --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
          --output '"$OUT"'/bc2_b${B}_sh$SH.jsonl --limit-states 4000 \
          --max-candidates 30 --only-dp-ids '"$OUT"'/$IDS \
          --budget $B --mode $MODE --top-k 8 $EXTRA \
          --shard-index $SH --shard-count 8 >> '"$OUT"'/bc2_sh$SH.log 2>&1 && break
        sleep 10
      done
    }
    run 1 full   bc2_ids_all.txt
    run 2 pruned bc2_ids_hard.txt
    run 2 pruned bc2_ids_easy.txt --arms-only
    echo "[$(date -Is)] bc2 分片 $SH 全部完成" >> '"$OUT"'/bc2_sh$SH.log
  ) &
  i=$((i+1))
done
wait
echo TILDE_BC2_DONE
'
