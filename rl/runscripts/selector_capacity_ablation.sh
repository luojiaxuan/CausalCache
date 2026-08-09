#!/bin/bash
# note (luojiaxuan): selector 容量升级的消融链(TODO B1 + B2 合并成一条命令)。
#
# **为什么是消融链而不是"直接上最强的那档"**:v4 的病因是欠拟合 —— 训练过的
# 421 个 winnable 态上界 100% 却只做对 51.1%。要修它有三个互不相同的怀疑对象,
# 一步全改就分不清是哪一项起了作用:
#   linear → mlp            隔离「每帧打分需要非线性」
#   mlp → mlp_pair          隔离「子集分需要帧间交互」(加性结构表达不了)
#   关 → 开 --use-context   隔离「打分函数得知道当前屏长什么样」
#     ← 这一项我认为嫌疑最大:因果注意力下候选帧排在当前屏之前、看不到它,
#       于是 v4 是在问"这帧有用吗"却没告诉模型"对哪一步有用"。
#   mean → mean_max         隔离「均值池化把关键区域抹平了」
#
# **先缓存特征再消融**:索引遍是 no_grad 的、输出与打分头无关,八种组合各过
# 一遍 8B 前向等于把同一批图算八遍。缓存一次(约 30 分钟)后每档降到分钟级。
#
# **判据是训练集排序准确率,不是留出集**。探针问的是"有没有任何模型能拟合
# 训练集";拟合不上就是容量/特征不足,留出集数字此时毫无意义 —— v4 的教训
# 正是只看了留出集 0.5828,没发现训练集本身就没拟合上。
# 报告里读 `train_rank_acc`(训练后**静态复测**,不是滑动平均:实测同一次
# 运行滑动 0.4688 / 静态 0.8281,差 36 个点,用滑动值会误判成容量不足)。
#
# 用法(容器内):GPU=5 bash /data/selector_capacity_ablation.sh
set -u
GPU=${GPU:-0}
D=/data
REPO=$D/osworld/CausalCache
OUT=${OUT:-$D/v5abl}
CACHE=$OUT/index_features.pt
LABELS=${LABELS:-$D/oracle/labels_all.jsonl}
MANIFEST=$D/oracle/agentnet_screening_manifest_ubuntu_v1.jsonl
IMAGES=$D/agentnet-frames-v3
MODEL=$D/models/GUI-Owl-1.5-8B-Instruct
INIT=${INIT:-$D/v4/run1/selector_bundle.pt}
export PYTHONPATH=$REPO/code:$REPO/rl/code
mkdir -p "$OUT"
cd "$REPO" || exit 1

if [ ! -f "$CACHE" ]; then
  echo "[$(date -Is)] 缓存索引遍特征(一次性,约 30 分钟)"
  CUDA_VISIBLE_DEVICES=$GPU python3 rl/code/scripts/cache_index_features.py \
    --labels "$LABELS" --manifest "$MANIFEST" --image-root "$IMAGES" \
    --model-dir "$MODEL" --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
    --output "$CACHE" --budget 2 > "$OUT/cache.log" 2>&1 || {
      echo "缓存失败,见 $OUT/cache.log"; exit 1; }
fi
echo "[$(date -Is)] 特征缓存就绪:$CACHE"

# 每档:名称|额外参数。lr 按参数量下调 —— mlp_pair 有 1310 万参数(v4 是 4097),
# 冒烟时 1e-3 下 mean_pair_loss 从 3.2 涨到 12.7,沿用 v4 的 lr 大概率发散。
run () {
  name=$1; shift
  [ -f "$OUT/$name/report.json" ] && { echo "[跳过] $name 已有结果"; return; }
  echo "[$(date -Is)] 训练 $name"
  CUDA_VISIBLE_DEVICES=$GPU python3 rl/code/scripts/train_selector_supervised.py \
    --labels "$LABELS" --manifest "$MANIFEST" --image-root "$IMAGES" \
    --model-dir "$MODEL" --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
    --init-head "$INIT" --output-root "$OUT/$name" --feature-cache "$CACHE" \
    --epochs 8 --report-every 400 "$@" > "$OUT/$name.log" 2>&1 \
    || echo "  ✗ $name 失败,见 $OUT/$name.log"
}

run linear            --head-arch linear   --pooling mean     --learning-rate 1e-3
run mlp               --head-arch mlp      --pooling mean     --learning-rate 3e-4
run mlp_ctx           --head-arch mlp      --pooling mean     --learning-rate 3e-4 --use-context
run mlp_ctx_maxpool   --head-arch mlp      --pooling mean_max --learning-rate 3e-4 --use-context
run pair_ctx          --head-arch mlp_pair --pooling mean     --learning-rate 1e-4 --use-context
run pair_ctx_maxpool  --head-arch mlp_pair --pooling mean_max --learning-rate 1e-4 --use-context
# 特征信息量探针(TODO B2):故意过拟合 —— 拟合得起来才说明特征里有信息
run probe_overfit     --head-arch mlp_pair --pooling mean_max --learning-rate 1e-4 \
                      --use-context --hidden 2048 --epochs 20 --holdout-frac 0.05

echo
echo "=== 消融汇总(先看 train_rank_acc:上不去就是容量/特征不足)==="
python3 - "$OUT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
print(f"{'配置':<20}{'参数':>10}{'训练集排序':>11}{'留出集排序':>11}{'过拟合门槛':>11}")
for d in sorted(root.iterdir()):
    f = d / "report.json"
    if not f.is_file():
        continue
    r = json.loads(f.read_text())
    print(f"{d.name:<20}{r.get('params',0):>10}"
          f"{r.get('train_rank_acc',0):>11.4f}{r.get('holdout_rank_acc',0):>11.4f}"
          f"{str(r.get('fit_gate_pass')):>11}")
print("\n读法:train_rank_acc 仍在 0.6 附近 = 该档仍欠拟合,别看留出集;"
      "\n      probe_overfit 都上不去 = 病在**特征**不在头,该改索引遍"
      "(分辨率/池化/取多层)而不是继续换头。")
PY
