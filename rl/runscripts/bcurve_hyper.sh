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
    # note (luojiaxuan): **B=2 也要重跑,不能复用 labels_all**,两个原因:
    #   ① labels_all 里的 easy 态走了 --skip-easy,只有一行 mode:screen、
    #      没有 oracle —— 而 easy 层正是这条曲线区分"头寸"与"净收益"的关键,
    #      复用等于把 400 个 easy 态全丢进过滤器(空跑归约时实测只剩 5 个);
    #   ② 今天把单帧探针的 top 集合改成含搭档帧,B=2 的子集池由 C(8,2)=28
    #      变成 C(9,2)=36,与旧产物不是同一口径。
    # 整条曲线必须出自同一份代码。顺序 1→2→4:先便宜的,B=4 最贵放最后。
    # note (luojiaxuan): 实测 2.91 前向/秒(5 卡),全量三档要 13.2 小时,太长。
    # 唯一不影响结论的砍法:**easy 态只打三臂、不枚举 oracle**(--arms-only)。
    # easy 层要回答的是"加历史图会不会把本来答对的题弄坏",而那由 recent-B
    # 这条可部署臂回答;easy 的 oracle 接近饱和,不值 48-83 次前向。
    # 跨预算的可部署对照仍是全量 1000 态,一点没缩。这样 13.2h → 约 8.5h。
    #
    # B=1 例外:它本来就便宜(约 13 次/态),整批 1000 态全枚举,
    # 顺带白拿一列 easy 层的 B=1 oracle 作参照。
    for spec in "1 full all" "2 pruned hard" "4 pruned hard"; do
      set -- $spec; B=$1; MODE=$2; SCOPE=$3
      if [ "$SCOPE" = "all" ]; then
        RUNS="$IDS|"
      else
        RUNS="$OUT/bcurve_ids_hard.txt|;$OUT/bcurve_ids_easy.txt|--arms-only"
      fi
      OLDIFS=$IFS; IFS=";"
      for run in $RUNS; do
        IFS=$OLDIFS
        ids=${run%%|*}; extra=${run#*|}
        for attempt in 1 2 3 4 5; do
          CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO/code:$REPO/rl/code \
            python3 "$REPO/rl/code/scripts/rl_oracle_enumerate.py" \
            --manifest "$MANIFEST" --image-root "$IMAGES" \
            --model-dir "$MODEL" --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
            --output "$OUT/bcurve_b${B}_sh$SH.jsonl" --limit-states "$LIMIT" \
            --max-candidates 30 --only-dp-ids "$ids" \
            --budget "$B" --mode "$MODE" --top-k 8 $extra \
            --shard-index "$SH" --shard-count "$SHARD_COUNT" >> "$LOG" 2>&1 && break
          echo "[$(date -Is)] B=$B($ids)分片 $SH 第 $attempt 次异常退出,10s 后续跑" >> "$LOG"
          sleep 10
        done
        IFS=";"
      done
      IFS=$OLDIFS
      echo "[$(date -Is)] B=$B 分片 $SH 结束,累计 $(wc -l < "$OUT/bcurve_b${B}_sh$SH.jsonl") 态" >> "$LOG"
    done
    echo "[$(date -Is)] 分片 $SH 全部完成" >> "$LOG"
  ) &
done
wait
echo "BCURVE_HOST_DONE"
