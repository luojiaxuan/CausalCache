#!/bin/bash
# note (luojiaxuan): B 曲线重测在 hyper 上的启动器(容器内执行)。
#
# 与 tilde_bcurve.sbatch 同一份实验,只是换了调度器:tilde 那边账号 GRES
# 硬顶 8 卡且已被同账号另一个项目的作业占满,作业只能排队,所以主力放 hyper。
# 两边**分片空间必须不重叠** —— tilde 作业已撤,这里用 SHARD_COUNT=5 独占;
# 日后 tilde 空出来要补投,按 dp_id 差集另出白名单,不要复用这套分片编号。
#
# 每卡串行跑 B=1(全枚举,便宜,先出曲线左端)再 B=4(剪枝 top-8,约 80 次
# 前向/态)。B=2 复用已有的 labels_all.jsonl 全枚举结果,不重跑。
#
# **supervisor 重启环**:共享机上跑长作业必须能自愈 —— 邻居 OOM、驱动抖动、
# 容器被别的会话重启都属常态。枚举脚本启动时读回自己输出里已完成的 dp_id,
# 所以重启的最大重放是一个 state;这里外面再套最多 5 次重试。
# 进度信号 = 各 jsonl 的行数(monitor 据此判断是否卡死)。
set -u
GPUS=${GPUS:?需给出 GPU id 列表,如 "6 7"}
SHARDS=${SHARDS:?需给出与 GPUS 等长的分片号列表}
SHARD_COUNT=${SHARD_COUNT:-5}
LIMIT=${LIMIT:-4000}
D=/data
REPO=$D/osworld/CausalCache
IMAGES=$D/agentnet-frames-v3
MANIFEST=$D/oracle/agentnet_screening_manifest_ubuntu_v1.jsonl
MODEL=$D/models/GUI-Owl-1.5-8B-Instruct
OUT=$D/oracle
IDS=$OUT/bcurve_ids.txt

[ -f "$IDS" ] || { echo "缺配对白名单 $IDS"; exit 1; }
[ -f "$MODEL/.snapshot.json" ] || { echo "缺 $MODEL/.snapshot.json(冻结守卫要)"; exit 1; }
cd "$REPO" || exit 1

set -- $GPUS
gpus=("$@")
set -- $SHARDS
shards=("$@")
[ ${#gpus[@]} -eq ${#shards[@]} ] || { echo "GPUS 与 SHARDS 长度不一致"; exit 1; }

for k in "${!gpus[@]}"; do
  G=${gpus[$k]}; SH=${shards[$k]}
  (
    LOG=$OUT/bcurve_sh$SH.log
    echo "[$(date -Is)] 起 GPU $G / 分片 $SH(共 $SHARD_COUNT)" >> "$LOG"
    for spec in "1 full" "4 pruned"; do
      set -- $spec; B=$1; MODE=$2
      for attempt in 1 2 3 4 5; do
        CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO/code:$REPO/rl/code \
          python3 "$REPO/rl/code/scripts/rl_oracle_enumerate.py" \
          --manifest "$MANIFEST" --image-root "$IMAGES" \
          --model-dir "$MODEL" --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
          --output "$OUT/bcurve_b${B}_sh$SH.jsonl" --limit-states "$LIMIT" \
          --max-candidates 30 --only-dp-ids "$IDS" \
          --budget "$B" --mode "$MODE" --top-k 8 \
          --shard-index "$SH" --shard-count "$SHARD_COUNT" >> "$LOG" 2>&1 && break
        echo "[$(date -Is)] B=$B 分片 $SH 第 $attempt 次异常退出,10s 后从断点续跑" >> "$LOG"
        sleep 10
      done
      echo "[$(date -Is)] B=$B 分片 $SH 结束,累计 $(wc -l < "$OUT/bcurve_b${B}_sh$SH.jsonl") 态" >> "$LOG"
    done
    echo "[$(date -Is)] 分片 $SH 全部完成" >> "$LOG"
  ) &
done
wait
echo "BCURVE_HOST_DONE"
