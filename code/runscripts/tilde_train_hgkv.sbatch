#!/bin/bash
#SBATCH --job-name=cc-hgkv
#SBATCH --partition=main
#SBATCH --gres=gpu:4
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
# note (luojiaxuan): **卡数是科学变量,不是资源参数。** 每优化步的全局批
# = gradient_accumulation_steps × micro_batch × world_size,所以 8 卡在同样的
# max_optimizer_steps 下见的数据是 4 卡的两倍(1340 单元:4 卡 300 步 ≈ 1.79 epoch,
# 8 卡 ≈ 3.58 epoch,且会先撞 epochs=3 的上限在约 251 步停)。
# 要与 hyper00 的 4 卡臂构成"只换种子"的复现,这里必须也是 4。
# 想用满 8 卡就得同时按比例改 max_optimizer_steps,并且那已经不是复现了。
NPROC=${NPROC:-4}
CONFIG=${CONFIG:-code/configs/causalcache_desktop_did_hgkv_v7_poolrank.json}
ROOT=$HOME/causalcache
REPO=$ROOT/CausalCache
CORPUS=$ROOT/v7pool/corpus
MODEL=$ROOT/models/GUI-Owl-1.5-8B-Instruct
OUT=$ROOT/runs/$VARIANT
IMAGE=$ROOT/sglang-omni-dev.sqsh

mkdir -p "$OUT"
# note (luojiaxuan): 冻结视觉运行时的身份校验要求模型目录带 .snapshot.json,
# 且解析后必须与 Git 清单相等(repo + revision + 14 个文件的 size/sha256)。
# 新拉的 HF 快照不带这个文件 —— 首次提交就是栽在这儿。这里做成幂等的:
# 缺了就从仓库补,补完仍由 runtime 自己逐文件核 sha,**不绕过校验**。
SNAP=$MODEL/.snapshot.json
[ -f "$SNAP" ] || cp "$REPO/code/configs/gui_owl_1_5_8b_snapshot.json" "$SNAP"
# 守卫还要求目录内容与清单**精确相等**(14 个文件 + .snapshot.json,不许有多余)。
# HF 下载会留下 .cache/、README.md、.gitattributes —— 第二次提交栽在这儿。
# 这里只做**检查并报错**,不在作业里删模型文件(那太危险);
# 修法写在报错里,由人执行一次即可,之后幂等。
python3 - "$MODEL" <<'PYCHK' || exit 2
import json, os, sys
root = sys.argv[1]
man = json.load(open(os.path.join(root, ".snapshot.json")))
allowed = {f["path"] for f in man["files"]} | {".snapshot.json"}
actual = {os.path.relpath(os.path.join(dp, fn), root)
          for dp, _dn, fns in os.walk(root) for fn in fns}
extra, missing = sorted(actual - allowed), sorted(allowed - actual)
if extra or missing:
    print(f"FATAL 模型目录清单不符 —— 多余 {len(extra)} 个、缺失 {len(missing)} 个")
    for r in (extra + missing)[:5]:
        print("   ", r)
    print("修法:删掉多余文件(常见是 .cache/、README.md、.gitattributes)后重提交")
    sys.exit(1)
print(f"模型清单校验通过:{len(actual)} 个文件")
PYCHK
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
       # note (luojiaxuan): 冻结视觉守卫钉死 transformers==5.6.0,并核对视觉模块的
       # **源码哈希**——不是只比版本号。tilde 的镜像 hongccc/sglang-omni:dev 带的是
       # 5.12.1(torch 相同),第三次提交栽在这儿。不改镜像,把 5.6.0 装到挂载目录
       # 再前置 PYTHONPATH;pylibs 必须排在 $REPO/code **前面**才能盖住镜像自带的。
       export PYTHONPATH=$ROOT/pylibs:$REPO/code
       export HF_HOME=$ROOT/hf
       torchrun --standalone --nproc_per_node=${NPROC:-4} code/scripts/train_success_sft_lora.py \
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
echo "=== exit $CODE $(date -u +%FT%TZ) ==="
[ $CODE -eq 0 ] && touch "$OUT/EXIT_0"
exit $CODE
