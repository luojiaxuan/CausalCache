#!/bin/bash
# note (luojiaxuan): OSWorld 15/30 步双臂重跑,消除跨主机混杂。
#
# 为什么重跑:原来各臂按"哪台机器空闲"分片,selector 臂有 1/3 任务落在 aries(A6000),
# 而对照臂一个都没有。aries 比三台现代卡低约 9pp(步数中位 25 vs 17,图形密集域塌陷),
# 足以凭空造出 -2.8pp 的假差异。
#
# 本次设计:
#   - 只用 hyper00 / hyper01(都是 H200),完全排除 aries;
#   - shard -> 主机的映射四轮固定不变,因此每个任务在四种条件下都在同一台机器上跑,
#     主机效应被配对差完全抵消;
#   - 每轮开始前重启服务器(见全局规则:推理服务器禁止跨臂复用,会泄漏到 TB 级)。
#
# 需要的环境变量见下方 : "${VAR:?}" 断言。
set -u
: "${HOSTTAG:?}"        # h00 / h01
: "${GPUS:?}"           # "0 1"
: "${SHARDS:?}"         # "0 1 2 3"
: "${BASE:?}"           # 宿主侧 osworld 根
: "${OSWROOT:?}"        # OSWorld 仓库
: "${RUNDIR:?}"         # worker 的 cwd
: "${WORKERCODE:?}"     # 宿主侧 CausalCache/code
: "${CREPO:?}"          # 容器内 repo
: "${CMODEL:?}"         # 容器内模型目录
: "${CSEL:?}"           # 容器内 selector bundle
CONT=sglang-omni-jaxan
SHARD_COUNT=12
PORT0=19300
LOG=$BASE/rerun/driver-$HOSTTAG.log
mkdir -p "$BASE/rerun" "$BASE/rerun/logs"
exec >>"$LOG" 2>&1 || { echo "FATAL: 日志不可写 $LOG" >&2; exit 1; }
echo "=== rerun driver $HOSTTAG 启动 $(date -Is) shards=[$SHARDS] gpus=[$GPUS] ==="

n_srv=0; for g in $GPUS; do n_srv=$((n_srv+2)); done   # 每卡两个副本

stop_servers() {
  docker exec $CONT bash -lc 'pkill -f "[o]sw_rerun_serve.sh"; sleep 2; pkill -f "[s]erve_osworld_official_policy"; true'
  sleep 12
}

start_servers() {   # $1 = sel|recent
  local arm=$1 i=0 p g
  docker exec $CONT bash -lc "cat > /tmp/osw_rerun_serve.sh <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
DEV=\$1; PORT=\$2; LOG=\$3; ARM=\$4
EXTRA=()
[ \"\$ARM\" = sel ] && EXTRA+=(--selector-bundle $CSEL --selector-arch two_tower --selector-beam 3)
cd $CREPO
while true; do
  CUDA_VISIBLE_DEVICES=\$DEV PYTHONPATH=code python3 code/scripts/serve_osworld_official_policy.py \\
    --model-dir $CMODEL --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \\
    --device cuda:0 --port \"\$PORT\" --visual-tokens 2560 --memory-budget 4 \"\${EXTRA[@]}\" >> \"\$LOG\" 2>&1
  echo REPLICA_EXIT_\$? >> \"\$LOG\"
  sleep 10
done
EOS
chmod +x /tmp/osw_rerun_serve.sh"
  # note (luojiaxuan): 错峰启动。8 个副本同时加载 8B 权重会在宿主上造成内存尖峰,
  # 2026-08-01 h01 的容器就是在这个当口被 SIGKILL(ExitCode 137、OOMKilled=false,
  # 即宿主级 OOM 杀 PID 1)。每个副本间隔 45s,把尖峰摊平。
  for g in $GPUS; do
    for _ in 1 2; do
      p=$((PORT0 + i)); i=$((i+1))
      docker exec -d $CONT bash -lc "bash /tmp/osw_rerun_serve.sh $g $p /tmp/rerun-$arm-$p.log $arm"
      sleep 45
    done
  done
  # 等全部就绪
  local k u
  for k in $(seq 1 60); do
    u=0; i=0
    for g in $GPUS; do for _ in 1 2; do
      p=$((PORT0 + i)); i=$((i+1))
      curl -sm 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 && u=$((u+1))
    done; done
    [ "$u" -eq "$n_srv" ] && break
    sleep 20
  done
  echo "$(date -Is) arm=$arm servers_up=$u/$n_srv"
  [ "$u" -eq "$n_srv" ]
}

run_round() {   # $1=arm  $2=steps
  local arm=$1 steps=$2 out tag pids=() s p i=0
  tag="${arm}-s${steps}"
  out=$BASE/rerun/out-$tag
  if [ -f "$BASE/rerun/DONE-$tag-$HOSTTAG" ]; then echo "skip $tag"; return 0; fi
  mkdir -p "$out"
  # note (luojiaxuan): 容器可能被外部杀掉(共享机常态)。每轮前确认存活,否则拉起来;
  # 拉不起来就中止本轮,不要空转四轮什么都不跑。
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONT"; then
    echo "$(date -Is) 容器 $CONT 不在运行,尝试重启"
    docker start "$CONT" >/dev/null 2>&1; sleep 10
    docker ps --format '{{.Names}}' | grep -qx "$CONT" || { echo "$(date -Is) $tag ABORT 容器拉不起来"; return 1; }
  fi
  stop_servers
  start_servers "$arm" || { echo "$(date -Is) $tag ABORT 服务器未就绪"; return 1; }
  echo "$(date -Is) === 开跑 $tag shards=[$SHARDS] ==="
  for s in $SHARDS; do
    p=$((PORT0 + (i % n_srv))); i=$((i+1))
    (
      cd "$RUNDIR"
      for a in 1 2 3 4 5; do
        PYTHONPATH=$WORKERCODE:$OSWROOT /usr/bin/python3 \
          $WORKERCODE/scripts/run_osworld_benchmark_worker.py \
          --osworld-root "$OSWROOT" --meta-path evaluation_examples/test_nogdrive.json \
          --shard-index "$s" --shard-count $SHARD_COUNT --output-root "$out" \
          --policy-endpoint "http://127.0.0.1:$p/act" \
          --memory-arm full --memory-budget 4 --max-steps "$steps" \
          --cache-dir "$BASE/rerun/cache-$tag" >> "$BASE/rerun/logs/$tag-shard$s.log" 2>&1
        code=$?
        last=$(grep -h '"event": "WORKER_COMPLETE"' "$BASE/rerun/logs/$tag-shard$s.log" | tail -1)
        if [ $code -eq 0 ] && echo "$last" | grep -q '"failed": 0'; then break; fi
        sleep 30
      done
    ) &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  local n; n=$(ls -1 "$out"/*/*/result.json 2>/dev/null | wc -l)
  echo "$(date -Is) === $tag 完成,本机累计结果 $n ==="
  touch "$BASE/rerun/DONE-$tag-$HOSTTAG"
}

# note (luojiaxuan): 先做 15 步双臂(论文主结果最需要),再做 30 步。
run_round sel    15
run_round recent 15
run_round sel    30
run_round recent 30
stop_servers
echo "=== rerun driver $HOSTTAG 全部完成 $(date -Is) ==="
