#!/bin/bash
# note (luojiaxuan): CausalCache-RL 迭代编排(hyper01 宿主侧运行)。
# 每迭代四阶段,阶段完成落 marker,重启从 marker 续(全局断点续跑规则):
#   R rollout(τ>0 采样)→ C collect(三键 join)→ T train(GRPO)→ 滚动重启
#
# 拓扑(h01 单机,容器 sglang-omni-jaxan-2,2026-08-06 起重建为 --gpus all,
# 容器序号 = 宿主序号):
#   rollout 阶段:S 个 policy server 各占 GPUS 列表一张卡;worker 在宿主(要 dockerd 起 VM)
#   train  阶段:server 已杀,trainer DDP(torchrun,episode 分片)占 TRAIN_GPUS
#   两阶段串行,峰值占用 = max(len(GPUS), len(TRAIN_GPUS)) ≤ 6 卡/机上限
#
# 参数化(env 覆盖):
#   ITER_START/ITER_END  迭代区间(含);TASKS_PER_ITER=8;G=6;WORKERS=4
#   GPUS="1 2 3"  server 用卡列表(空格分隔容器序号;共享机上按实际空闲卡传);
#                 SERVERS 默认 = 列表长度;TRAIN_GPUS 默认 = GPUS 全列表
#                 (trainer DDP 按 episode 分片,train 段时长 ≈ 单卡 / 卡数)
#   MAX_STEPS=30(训练 rollout cap;评测另走 50 步协议)
#   TAU=1.0(Plackett-Luce 温度);BUDGET=2(用户裁定 B=2)
#   HEALTH_TIMEOUT=300  server 健康轮询窗口秒数(15s 一轮;iter-8 教训:
#                       单次 110s 定时检查太脆,重载稍慢即误判死亡)
#   EVAL_EVERY=20(到期只落 EVAL_DUE marker,held-out 评测走独立脚本)
#
# 已固化的坑(全部踩过):worker cwd 必须是 run30(否则重下 11.4G VM 镜像);
# endpoint 用容器 IP(bridge);pkill 模式必须 [x] 括号;server 每迭代重启(RAM 泄漏)。
set -u
ITER_START=${ITER_START:?}; ITER_END=${ITER_END:?}
TASKS_PER_ITER=${TASKS_PER_ITER:-8}; G=${G:-6}
WORKERS=${WORKERS:-4}
GPUS=${GPUS:-"1 2 3"}; GPU_ARR=($GPUS)
SERVERS=${SERVERS:-${#GPU_ARR[@]}}
[ "$SERVERS" -le "${#GPU_ARR[@]}" ] || { echo "SERVERS=$SERVERS 超过 GPUS 列表长度 ${#GPU_ARR[@]}"; exit 1; }
TRAIN_GPUS=${TRAIN_GPUS:-$GPUS}; TG_ARR=($TRAIN_GPUS)
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-300}
MAX_STEPS=${MAX_STEPS:-30}; TAU=${TAU:-1.0}; BUDGET=${BUDGET:-2}
# selector v2 为默认:hidden = 策略 hidden-state 打分头(零手写特征);
# cheap = 旧 28 维两塔,仅作对照臂。iter-0 bootstrap 产物不同:
#   hidden → make_hidden_head.py 的头 bundle;cheap → v4 marginal_scorer.pt
SELECTOR_MODE=${SELECTOR_MODE:-hidden}
EVAL_EVERY=${EVAL_EVERY:-20}

# 主机档案(默认 = h01;h00 用 env 覆盖:CTN=sglang-omni-jaxan RLC=/data/rl
#   REPO=/data/osworld/CausalCache WREPO=/data04/jaxan/osworld/CausalCache
#   MODEL=/data/models/GUI-Owl-1.5-8B-Instruct,B/RLH/CACHE 两机恰好同值)
CTN=${CTN:-sglang-omni-jaxan-2}
B=${B:-/data04/jaxan/osworld}           # 宿主视角
RLH=${RLH:-/data04/jaxan/rl}            # 宿主视角迭代根
RLC=${RLC:-/bigdata/rl}                 # 容器视角同一目录
REPO=${REPO:-/bigdata/osworld/CausalCache}  # 容器视角仓库
WREPO=${WREPO:-/data02/jaxan/CausalCache-mwhgkv}  # 宿主视角 worker 仓库
CACHE=${CACHE:-$B/cache-fast}           # worker 任务配置缓存
BAND=${BAND:-$RLC/osworld_rl_train_band_v1.json} # 可学带(env 可覆盖;v2 起裁掉 30 步 cap 下持续全败的任务)
MODEL=${MODEL:-/bigdata/models/GUI-Owl-1.5-8B-Instruct}
PORT0=19511

log() { echo "[$(date -u +%FT%TZ)] iter=$ITER $*"; }
cexec() { docker exec "$CTN" bash -lc "$*"; }

CIP=$(docker inspect "$CTN" --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
# host 网络容器 inspect 不给 IP(h00 即此形态),回退 127.0.0.1;bridge(h01)不受影响
case "$CIP" in ""|*[!0-9.]*) CIP=127.0.0.1;; esac

for ITER in $(seq "$ITER_START" "$ITER_END"); do
  PREV=$((ITER-1))
  ID_H=$RLH/iter-$ITER; ID_C=$RLC/iter-$ITER
  PD_C=$RLC/iter-$PREV
  cexec "mkdir -p $ID_C/audit && chmod -R 777 $ID_C"

  # ---------- 0. 本迭代任务抽样(确定性:seed=迭代号) ----------
  if [ ! -f "$ID_H/tasks.json" ]; then
    cexec "python3 - <<PY
import json, random
band = json.load(open('$BAND'))
flat = [(d, t) for d in sorted(band) for t in band[d]]
random.Random(20260804 + $ITER).shuffle(flat)
picked = flat[:$TASKS_PER_ITER]
meta = {}
for d, t in picked: meta.setdefault(d, []).append(t)
json.dump(meta, open('$ID_C/tasks.json', 'w'), indent=1)
print('iter $ITER 任务:', sum(len(v) for v in meta.values()))
PY"
    cp "$ID_H/tasks.json" "$B/OSWorld/evaluation_examples/rl_iter_$ITER.json" 2>/dev/null \
      || cexec "cp $ID_C/tasks.json /bigdata/osworld/OSWorld/evaluation_examples/rl_iter_$ITER.json" || true
    # 宿主视角的 OSWorld 也要一份(worker 用宿主路径)
    [ -f "$B/OSWorld/evaluation_examples/rl_iter_$ITER.json" ] \
      || cp "$ID_H/tasks.json" "$B/OSWorld/evaluation_examples/rl_iter_$ITER.json"
  fi

  launch_servers() {  # 参数:$1 = fatal|soft(健康检查失败是否致命)
    SEL=$PD_C/selector_bundle.pt
    if [ "$SELECTOR_MODE" = "hidden" ]; then
      SEL_FLAGS="--selector-mode hidden --selector-head $SEL"
    else
      SEL_FLAGS="--selector-bundle $SEL --selector-arch two_tower"
    fi
    ADP_FLAGS=""
    if cexec "test -f $PD_C/adapter.pt"; then
      SHA=$(cexec "sha256sum $PD_C/adapter.pt | cut -d' ' -f1")
      ADP_FLAGS="--adapter-checkpoint $PD_C/adapter.pt --adapter-checkpoint-sha256 $SHA"
    fi
    # expandable_segments:8 worker 并发下 serve 端出现过瞬时 18-20G 分配的
    # 碎片型 OOM(iter-9,64min 内 8 次,修复遍自愈);该开关显著降低碎片失败率
    for s in $(seq 0 $((SERVERS-1))); do
      docker exec -d "$CTN" bash -lc "cd $REPO && CUDA_VISIBLE_DEVICES=${GPU_ARR[$s]} \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=code \
        python3 rl/code/scripts/serve_osworld_rl_policy.py \
        --model-dir $MODEL --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
        --device cuda:0 --port $((PORT0+s)) --visual-tokens 2560 \
        --memory-budget $BUDGET \
        $SEL_FLAGS \
        --selector-temperature $TAU --rl-audit-dir $ID_C/audit \
        $ADP_FLAGS >> $ID_C/serve-$s.log 2>&1"
    done
    # 健康轮询窗口(替代单次定时检查):HEALTH_TIMEOUT 秒内每 15s 全量探活,
    # 全部 200 即通过;窗口耗尽仍有掉队者才按 fatal|soft 处置
    local t0=$SECONDS ok=0
    while [ $((SECONDS - t0)) -lt "$HEALTH_TIMEOUT" ]; do
      ok=1
      for s in $(seq 0 $((SERVERS-1))); do
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://$CIP:$((PORT0+s))/health" --max-time 5)
        [ "$code" = "200" ] || { ok=0; break; }
      done
      [ "$ok" = "1" ] && break
      sleep 15
    done
    if [ "$ok" != "1" ]; then
      for s in $(seq 0 $((SERVERS-1))); do
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://$CIP:$((PORT0+s))/health" --max-time 5)
        log "server $s(GPU ${GPU_ARR[$s]}) health=$code(窗口 ${HEALTH_TIMEOUT}s 耗尽)"
      done
      [ "$1" = "fatal" ] && exit 1
    fi
  }

  run_generation() {  # 参数:$1 = 代号 g
    local g=$1
    cexec "mkdir -p $ID_C/out-g$g && chmod 777 $ID_C/out-g$g"
    for s in $(seq 0 $((WORKERS-1))); do
      ( cd "$B/run30" && PYTHONPATH=$WREPO/code:$B/OSWorld /usr/bin/python3 \
          "$WREPO/code/scripts/run_osworld_benchmark_worker.py" \
          --osworld-root "$B/OSWorld" \
          --meta-path "evaluation_examples/rl_iter_$ITER.json" \
          --shard-index "$s" --shard-count "$WORKERS" \
          --output-root "$RLH/iter-$ITER/out-g$g" \
          --policy-endpoint "http://$CIP:$((PORT0 + s % SERVERS))/act" \
          --memory-arm full --memory-budget "$BUDGET" \
          --max-steps "$MAX_STEPS" \
          --cache-dir "$CACHE" \
          >> "$ID_H/worker-g$g-$s.log" 2>&1 ) &
    done
    wait
  }

  # ---------- 1. rollout ----------
  if [ ! -f "$ID_H/ROLLOUT_DONE" ]; then
    log "R: 启动 $SERVERS 个 server(τ=$TAU B=$BUDGET)"
    launch_servers fatal
    log "R: $G 代 × $WORKERS worker rollout"
    for g in $(seq 0 $((G-1))); do
      run_generation "$g"
      log "R: 代 $g 完成"
    done
    # ---- 修复遍(2026-08-06 iter-6 实测):cuDNN mha_graph 逐请求失败会毒化
    # server —— 进程活着、/health 200,但请求全 500,挂它的 worker 全程 HTTPError
    # (iter-6 损失 18/48)。判据 = serve 日志 OSWORLD_POLICY_FAILURE 计数;
    # >0 则全量重启 server 后把所有代补跑一遍(断点续跑令已完成任务瞬时跳过)。
    PF=$(cexec "cat $ID_C/serve-*.log 2>/dev/null | grep -c OSWORLD_POLICY_FAILURE" | tr -cd '0-9')
    if [ "${PF:-0}" -gt 0 ]; then
      log "R: 检测到 $PF 次策略失败,重启 server 补跑全部代"
      cexec "pkill -f '[s]erve_osworld_rl_policy' || true"; sleep 5
      launch_servers soft
      for g in $(seq 0 $((G-1))); do run_generation "$g"; done
      log "R: 补跑完成"
    fi
    # 滚动重启纪律:rollout 一结束立刻杀 server(监督循环没有,直接杀进程)
    cexec "pkill -f '[s]erve_osworld_rl_policy' || true"; sleep 3
    n=$(find "$ID_H"/out-g* -name result.json 2>/dev/null | wc -l)
    log "R: 共 $n 条 episode,server 已回收"
    touch "$ID_H/ROLLOUT_DONE"
  fi

  # ---------- 2. collect ----------
  if [ ! -f "$ID_H/COLLECT_DONE" ]; then
    ROOTS=""
    for g in $(seq 0 $((G-1))); do ROOTS="$ROOTS --rollout-root $ID_C/out-g$g"; done
    cexec "cd $REPO && PYTHONPATH=code:rl/code python3 rl/code/scripts/collect_rl_trajectories.py \
      $ROOTS --audit-dir $ID_C/audit --output $ID_C/groups.jsonl" | tail -1
    touch "$ID_H/COLLECT_DONE"
  fi

  # ---------- 3. train ----------
  if [ ! -f "$ID_H/TRAIN_DONE" ]; then
    log "T: GRPO(DDP ×${#TG_ARR[@]} @ GPU $TRAIN_GPUS)"
    RES=""
    cexec "test -f $PD_C/adapter.pt" && RES="--resume-adapter $PD_C/adapter.pt"
    TG_CSV=${TRAIN_GPUS// /,}
    cexec "cd $REPO && CUDA_VISIBLE_DEVICES=$TG_CSV PYTHONPATH=code:rl/code \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 -m torch.distributed.run --standalone --nproc-per-node=${#TG_ARR[@]} \
      rl/code/scripts/train_causalcache_rl_grpo.py \
      --groups $ID_C/groups.jsonl \
      --model-dir $MODEL \
      --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
      --selector-bundle $PD_C/selector_bundle.pt $RES \
      --adapter-config code/configs/causalcache_desktop_did_hgkv_v7_poolrank.json \
      --max-groups-per-step $TASKS_PER_ITER \
      --seed $((20260804 + ITER)) \
      --output-root $ID_C 2>&1 | grep -vE 'processor_kwargs|Loading weights' | tail -3"
    cexec "test -f $ID_C/selector_bundle.pt && test -f $ID_C/adapter.pt" \
      || { log "FATAL train 产物缺失"; exit 1; }
    touch "$ID_H/TRAIN_DONE"
    log "T: 完成,iter_report: $(cexec "cat $ID_C/iter_report.json" | tr -d '\n')"
  fi

  # ---------- 4. 评测到期标记 ----------
  if [ $((ITER % EVAL_EVERY)) -eq 0 ]; then
    touch "$ID_H/EVAL_DUE"
    log "EVAL_DUE:held-out 120 任务 τ=0/50 步/2 轮,走独立评测脚本"
  fi
done
echo "LOOP_DONE $ITER_START..$ITER_END"
