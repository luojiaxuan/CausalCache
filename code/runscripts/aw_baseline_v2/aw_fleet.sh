#!/bin/bash
# note (luojiaxuan): stock AndroidWorld emulator 舰队(awextend_fleet.sh 的无 overlay 版)。
# 改动:去 overlay 挂载(stock 116 模板在镜像里)、探针换 stock 任务、支持 START 便于分批。
set -u
ACTION="${1:-up}"; N="${2:-29}"; BASE_PORT="${3:-42001}"; START="${4:-0}"
STAMP_FILE=/data04/jaxan/awfleet/fleet_stamp.txt
IMAGE=jaxanluo/sglang-omni:env
PROBE_TASK=SystemWifiTurnOn
name_for() { printf "sglang-omni-jaxan-%se%03d" "$1" "$2"; }
case "$ACTION" in
up)
  [ -f "$STAMP_FILE" ] || date +%m%d%H%M > "$STAMP_FILE"
  STAMP=$(cat "$STAMP_FILE")
  for i in $(seq "$START" $((START+N-1))); do
    port=$((BASE_PORT+i)); name=$(name_for "$STAMP" "$i")
    docker run -d --name "$name" --device /dev/kvm \
      -p "127.0.0.1:${port}:5000" --cpus 4 --memory 10g \
      "$IMAGE" >/dev/null 2>&1 || echo "[fleet] FAILED $name" >&2
  done
  echo "[fleet] launched idx $START..$((START+N-1)) stamp=$STAMP"
  ;;
wait)
  STAMP=$(cat "$STAMP_FILE"); TOTAL="$N"
  deadline=$(( $(date +%s) + 3000 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    ready=0
    for i in $(seq 0 $((TOTAL-1))); do
      port=$((BASE_PORT+i))
      curl -s -m 4 "http://127.0.0.1:${port}/suite/task_length?task_type=${PROBE_TASK}" 2>/dev/null | grep -q '"length"' && ready=$((ready+1))
    done
    echo "[fleet] ready=$ready/$TOTAL"
    [ "$ready" -eq "$TOTAL" ] && { echo ALL_READY; exit 0; }
    sleep 30
  done
  echo "[fleet] TIMEOUT ready=$ready/$TOTAL" >&2; exit 1
  ;;
urls)
  for i in $(seq 0 $((N-1))); do
    port=$((BASE_PORT+i))
    curl -s -m 4 "http://127.0.0.1:${port}/suite/task_length?task_type=${PROBE_TASK}" 2>/dev/null | grep -q '"length"' && echo "http://127.0.0.1:${port}"
  done
  ;;
down)
  names=$(docker ps -a --filter "name=sglang-omni-jaxan-" --format '{{.Names}}' | grep -E 'e[0-9]+$' || true)
  [ -z "$names" ] && { echo "[fleet] nothing"; exit 0; }
  echo "$names" | xargs -r docker rm -f >/dev/null 2>&1
  echo "[fleet] removed $(echo "$names" | wc -l)"
  ;;
esac
