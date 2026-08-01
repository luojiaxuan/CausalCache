#!/bin/bash
# note (luojiaxuan): 只补 sel@30 里被 aries(A6000)污染的 110 个任务,在 H200 上重跑。
# 不重跑其余部分:h100 的 251 个本来就干净;recent@30 与 15 步两臂从未碰过 aries;
# 且 h00/h01 已验证等价(106 个重复任务 98 个一致,独赢 5:3)。
set -u
: "${HOSTTAG:?}"; : "${GPUS:?}"; : "${SHARDS:?}"; : "${SHARD_COUNT:?}"
: "${BASE:?}"; : "${OSWROOT:?}"; : "${RUNDIR:?}"; : "${WORKERCODE:?}"
: "${CREPO:?}"; : "${CMODEL:?}"; : "${CSEL:?}"
CONT=sglang-omni-jaxan
PORT0=19400
OUT=$BASE/aries-repair/out-sel-s30
META=$BASE/aries-repair/meta.json
LOG=$BASE/aries-repair/driver-$HOSTTAG.log
mkdir -p "$OUT" "$BASE/aries-repair/logs"
exec >>"$LOG" 2>&1 || { echo "FATAL 日志不可写" >&2; exit 1; }
echo "=== aries 补跑 $HOSTTAG 启动 $(date -Is) shards=[$SHARDS]/$SHARD_COUNT gpus=[$GPUS] ==="

docker ps --format '{{.Names}}' | grep -qx "$CONT" || { docker start "$CONT" >/dev/null 2>&1; sleep 10; }

docker exec $CONT bash -lc "cat > /tmp/osw_repair_serve.sh <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
DEV=\$1; PORT=\$2; LOG=\$3
cd $CREPO
while true; do
  CUDA_VISIBLE_DEVICES=\$DEV PYTHONPATH=code python3 code/scripts/serve_osworld_official_policy.py \\
    --model-dir $CMODEL --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \\
    --device cuda:0 --port \"\$PORT\" --visual-tokens 2560 --memory-budget 4 \\
    --selector-bundle $CSEL --selector-arch two_tower --selector-beam 3 >> \"\$LOG\" 2>&1
  echo REPLICA_EXIT_\$? >> \"\$LOG\"
  sleep 10
done
EOS
chmod +x /tmp/osw_repair_serve.sh"

# note (luojiaxuan): 错峰启动,避免多副本同时加载权重造成宿主内存尖峰(h01 曾因此被 OOM 杀掉容器)。
i=0; n_srv=0
for g in $GPUS; do for _ in 1 2; do
  p=$((PORT0 + i)); i=$((i+1)); n_srv=$((n_srv+1))
  docker exec -d $CONT bash -lc "bash /tmp/osw_repair_serve.sh $g $p /tmp/repair-$p.log"
  sleep 45
done; done

for k in $(seq 1 60); do
  u=0; for j in $(seq 0 $((n_srv-1))); do
    curl -sm 3 "http://127.0.0.1:$((PORT0+j))/health" >/dev/null 2>&1 && u=$((u+1))
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
        --osworld-root "$OSWROOT" --meta-path "$META" \
        --shard-index "$s" --shard-count "$SHARD_COUNT" --output-root "$OUT" \
        --policy-endpoint "http://127.0.0.1:$p/act" \
        --memory-arm full --memory-budget 4 --max-steps 30 \
        --cache-dir "$BASE/aries-repair/cache" \
        >> "$BASE/aries-repair/logs/shard$s.log" 2>&1
      code=$?
      last=$(grep -h '"event": "WORKER_COMPLETE"' "$BASE/aries-repair/logs/shard$s.log" | tail -1)
      if [ $code -eq 0 ] && echo "$last" | grep -q '"failed": 0'; then break; fi
      sleep 30
    done
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "$(date -Is) === $HOSTTAG 完成,本机结果 $(ls -1 $OUT/*/*/result.json 2>/dev/null | wc -l) ==="
docker exec $CONT bash -lc 'pkill -f "[o]sw_repair_serve.sh"; sleep 2; pkill -f "[s]erve_osworld_official_policy"; true'
