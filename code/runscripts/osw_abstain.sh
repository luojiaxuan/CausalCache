#!/bin/bash
# note (luojiaxuan): 同域决定性实验——OSWorld 15 步,selector 开启弃权(τ=0)。
#
# 背景:同域五臂里 CausalCache-LA 34.07% vs Recent-4 32.69%,+1.11pp 不显著,
# 不一致对 25:21。按域拆分,损失集中在单应用文档类(writer 0赢4输、impress 2:4、
# thunderbird 0:2),收益集中在跨应用/图形类(multi_apps/gimp/os/vlc)。
# 说明不是"没头寸",是模型在没有值得提升的帧时仍被迫提升,挤掉 recent。
#
# 本臂唯一改动:--selector-min-marginal 0 —— 只提升预测边际 > 0 的候选,
# 不足 B 个用最近帧补齐。预算仍是 B=4。对照沿用已有的 recent-B4 全量 361。
set -u
: "${HOSTTAG:?}"; : "${GPUS:?}"; : "${SHARDS:?}"; : "${SHARD_COUNT:?}"
: "${BASE:?}"; : "${OSWROOT:?}"; : "${RUNDIR:?}"; : "${WORKERCODE:?}"
: "${CREPO:?}"; : "${CMODEL:?}"; : "${CSEL:?}"
CONT=sglang-omni-jaxan
PORT0=19500
OUT=$BASE/abstain/out-sel-tau0-s15
LOG=$BASE/abstain/driver-$HOSTTAG.log
mkdir -p "$OUT" "$BASE/abstain/logs"
exec >>"$LOG" 2>&1 || { echo "FATAL 日志不可写" >&2; exit 1; }
echo "=== 弃权臂 $HOSTTAG 启动 $(date -Is) shards=[$SHARDS]/$SHARD_COUNT gpus=[$GPUS] ==="

docker ps --format '{{.Names}}' | grep -qx "$CONT" || { docker start "$CONT" >/dev/null 2>&1; sleep 10; }
pkill -f "[r]un_osworld_benchmark_worker" 2>/dev/null && sleep 3

NETMODE=$(docker inspect "$CONT" --format '{{.HostConfig.NetworkMode}}')
if [ "$NETMODE" = host ]; then EPHOST=127.0.0.1
else EPHOST=$(docker inspect "$CONT" --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'); fi
[ -n "$EPHOST" ] || { echo "FATAL 取不到容器 IP"; exit 1; }
echo "$(date -Is) 网络=$NETMODE 端点=$EPHOST"

docker exec $CONT bash -lc "cat > /tmp/osw_abstain_serve.sh <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
DEV=\$1; PORT=\$2; LOG=\$3
cd $CREPO
while true; do
  CUDA_VISIBLE_DEVICES=\$DEV PYTHONPATH=code python3 code/scripts/serve_osworld_official_policy.py \\
    --model-dir $CMODEL --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \\
    --device cuda:0 --port \"\$PORT\" --visual-tokens 2560 --memory-budget 4 \\
    --selector-bundle $CSEL --selector-arch two_tower --selector-beam 3 \\
    --selector-min-marginal 0 >> \"\$LOG\" 2>&1
  echo REPLICA_EXIT_\$? >> \"\$LOG\"
  sleep 10
done
EOS
chmod +x /tmp/osw_abstain_serve.sh"

i=0; n_srv=0
for g in $GPUS; do for _ in 1 2; do
  p=$((PORT0 + i)); i=$((i+1)); n_srv=$((n_srv+1))
  docker exec -d $CONT bash -lc "bash /tmp/osw_abstain_serve.sh $g $p /tmp/abstain-$p.log"
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
        --cache-dir "$BASE/abstain/cache" \
        >> "$BASE/abstain/logs/shard$s.log" 2>&1
      code=$?
      last=$(grep -h '"event": "WORKER_COMPLETE"' "$BASE/abstain/logs/shard$s.log" | tail -1)
      if [ $code -eq 0 ] && echo "$last" | grep -q '"failed": 0'; then break; fi
      sleep 30
    done
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "$(date -Is) === $HOSTTAG 完成,本机结果 $(ls -1 $OUT/*/*/result.json 2>/dev/null | wc -l) ==="
docker exec $CONT bash -lc 'pkill -f "[o]sw_abstain_serve.sh"; sleep 2; pkill -f "[s]erve_osworld_official_policy"; true'
