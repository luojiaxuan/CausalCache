#!/bin/bash
# note (luojiaxuan): 通用 OSWorld 单臂 runner(15 步)。用于同批双臂对照:
#   ARMNAME=selfix  → 修复后的 selector(新 bundle + 截距 + τ 弃权)
#   ARMNAME=recent  → 同批 Recent-4 对照(不挂 selector)
# 为什么要同批对照:实测两个 98% 行为相同的臂,任务级仍翻转 14.4%、净差 -3.88pp。
# 跨轮比较在这个噪声地板下没有分辨力,必须同一时间窗、同一批机器跑对照。
set -u
: "${HOSTTAG:?}"; : "${GPUS:?}"; : "${SHARDS:?}"; : "${SHARD_COUNT:?}"
: "${BASE:?}"; : "${OSWROOT:?}"; : "${RUNDIR:?}"; : "${WORKERCODE:?}"
: "${CREPO:?}"; : "${CMODEL:?}"; : "${ARMNAME:?}"; : "${PORT0:?}"
CSEL="${CSEL:-}"; TAU="${TAU:-}"; OFFSET="${OFFSET:-0}"
CONT=sglang-omni-jaxan
OUT=$BASE/twoarm-r2/out-$ARMNAME
LOG=$BASE/twoarm-r2/driver-$ARMNAME-$HOSTTAG.log
mkdir -p "$OUT" "$BASE/twoarm-r2/logs"
exec >>"$LOG" 2>&1 || { echo "FATAL 日志不可写" >&2; exit 1; }
echo "=== $ARMNAME @$HOSTTAG 启动 $(date -Is) shards=[$SHARDS]/$SHARD_COUNT gpus=[$GPUS] tau=$TAU offset=$OFFSET ==="

docker ps --format '{{.Names}}' | grep -qx "$CONT" || { docker start "$CONT" >/dev/null 2>&1; sleep 10; }

NETMODE=$(docker inspect "$CONT" --format '{{.HostConfig.NetworkMode}}')
if [ "$NETMODE" = host ]; then EPHOST=127.0.0.1
else EPHOST=$(docker inspect "$CONT" --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'); fi
[ -n "$EPHOST" ] || { echo "FATAL 取不到容器 IP"; exit 1; }
echo "$(date -Is) 网络=$NETMODE 端点=$EPHOST"

SELARGS=""
[ -n "$CSEL" ] && SELARGS="--selector-bundle $CSEL --selector-arch two_tower --selector-beam 3 --selector-score-offset $OFFSET"
[ -n "$CSEL" ] && [ -n "$TAU" ] && SELARGS="$SELARGS --selector-min-marginal $TAU"

docker exec $CONT bash -lc "cat > /tmp/osw_arm_${ARMNAME}.sh <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
DEV=\$1; PORT=\$2; LOG=\$3
cd $CREPO
while true; do
  CUDA_VISIBLE_DEVICES=\$DEV PYTHONPATH=code python3 code/scripts/serve_osworld_official_policy.py \\
    --model-dir $CMODEL --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \\
    --device cuda:0 --port \"\$PORT\" --visual-tokens 2560 --memory-budget 4 \\
    $SELARGS >> \"\$LOG\" 2>&1
  echo REPLICA_EXIT_\$? >> \"\$LOG\"
  sleep 10
done
EOS
chmod +x /tmp/osw_arm_${ARMNAME}.sh"

i=0; n_srv=0
for g in $GPUS; do for _ in 1 2; do
  p=$((PORT0 + i)); i=$((i+1)); n_srv=$((n_srv+1))
  docker exec -d $CONT bash -lc "bash /tmp/osw_arm_${ARMNAME}.sh $g $p /tmp/arm-${ARMNAME}-$p.log"
  sleep 40
done; done

for k in $(seq 1 60); do
  u=0; for j in $(seq 0 $((n_srv-1))); do
    curl -sm 3 "http://$EPHOST:$((PORT0+j))/health" >/dev/null 2>&1 && u=$((u+1))
  done
  [ "$u" -eq "$n_srv" ] && break
  sleep 20
done
echo "$(date -Is) servers_up=$u/$n_srv"
[ "$u" -eq "$n_srv" ] || { echo "ABORT 服务器未就绪"; exit 1; }

pids=(); i=0
for s in $SHARDS; do
  p=$((PORT0 + (i % n_srv))); i=$((i+1))
  (
    cd "$RUNDIR"
    for a in 1 2 3 4 5; do
      PYTHONPATH=$WORKERCODE:$OSWROOT /usr/bin/python3 \
        $WORKERCODE/scripts/run_osworld_benchmark_worker.py \
        --osworld-root "$OSWROOT" --meta-path evaluation_examples/test_nogdrive.json \
        --shard-index "$s" --shard-count "$SHARD_COUNT" --output-root "$OUT" \
        --policy-endpoint "http://$EPHOST:$p/act" \
        --memory-arm full --memory-budget 4 --max-steps 15 \
        --cache-dir "$BASE/twoarm-r2/cache-$ARMNAME" \
        >> "$BASE/twoarm-r2/logs/$ARMNAME-shard$s.log" 2>&1
      code=$?
      last=$(grep -h '"event": "WORKER_COMPLETE"' "$BASE/twoarm-r2/logs/$ARMNAME-shard$s.log" | tail -1)
      if [ $code -eq 0 ] && echo "$last" | grep -q '"failed": 0'; then break; fi
      sleep 30
    done
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "$(date -Is) === $ARMNAME @$HOSTTAG 完成,本机 $(ls -1 $OUT/*/*/result.json 2>/dev/null | wc -l) ==="
# note (luojiaxuan): 必须连 python 服务器一起杀。只杀监督循环会留下孤儿服务器
# 继续占卡占内存——本轮两台共 12 个服务器、12 张卡就是这么漏的。
docker exec $CONT bash -lc "pkill -f '[o]sw_arm_${ARMNAME}.sh'; sleep 2; pkill -f '[s]erve_osworld_official_policy'; sleep 5; echo 剩余服务器=\$(pgrep -fc '[s]erve_osworld_official_policy' || echo 0)"
