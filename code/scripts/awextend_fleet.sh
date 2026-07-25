#!/bin/bash
# AW-Extend emulator fleet launcher.
#
# note (luojiaxuan): emulator 是 CPU 侧的(-gpu off / swiftshader 软渲染),GPU 只服务
# policy。所以"每个 GPU 并发多个模拟器"= 一 GPU 一个 policy 进程 + 多线程各驱动一台
# emulator。env 镜像把 console/grpc/bind 端口写死在容器内(5554/8554/5000),一容器
# 只能一台 emulator,所以按台起容器、只映射 5000 到不同主机端口。
#
# 这些是"一次性 fan-out"容器:run 结束立刻 docker rm -f,不复用。
set -u

ACTION="${1:-up}"
N="${2:-38}"
BASE_PORT="${3:-41001}"
STAMP_FILE=/data02/jaxan/staging/awextend/fleet_stamp.txt
IMAGE=jaxanluo/sglang-omni:env
OVERLAY=/data02/jaxan/staging/awextend/overlay

case "$ACTION" in
up)
  STAMP=$(date +%m%d%H%M)
  echo "$STAMP" > "$STAMP_FILE"
  mount_args=""
  if [ -d "$OVERLAY/android_world" ]; then
    mount_args="-v $OVERLAY/android_world:/android_world"
    echo "[fleet] mounting AW-Extend overlay from $OVERLAY/android_world"
  else
    echo "[fleet] WARNING: no overlay at $OVERLAY/android_world — stock AndroidWorld only" >&2
  fi
  for i in $(seq 0 $((N-1))); do
    port=$((BASE_PORT+i))
    name=$(printf "sglang-omni-jaxan-%se%02d" "$STAMP" "$i")
    docker run -d --name "$name" \
      --device /dev/kvm \
      -p "127.0.0.1:${port}:5000" \
      --cpus 4 --memory 10g \
      $mount_args \
      "$IMAGE" >/dev/null 2>&1 \
      && echo "[fleet] started $name -> 127.0.0.1:$port" \
      || echo "[fleet] FAILED to start $name" >&2
  done
  echo "[fleet] launched $N containers, stamp=$STAMP"
  ;;

wait)
  STAMP=$(cat "$STAMP_FILE")
  deadline=$(( $(date +%s) + 1800 ))
  ready=0
  while [ "$(date +%s)" -lt "$deadline" ]; do
    ready=0; dead=0
    for i in $(seq 0 $((N-1))); do
      port=$((BASE_PORT+i))
      name=$(printf "sglang-omni-jaxan-%se%02d" "$STAMP" "$i")
      st=$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || echo missing)
      if [ "$st" != "running" ]; then dead=$((dead+1)); continue; fi
      if curl -s -m 3 -o /dev/null "http://127.0.0.1:${port}/docs" 2>/dev/null; then
        ready=$((ready+1))
      fi
    done
    echo "[fleet] ready=$ready/$N dead=$dead"
    [ "$ready" -eq "$N" ] && { echo "[fleet] ALL_READY"; exit 0; }
    [ "$dead" -gt 0 ] && echo "[fleet] WARNING $dead container(s) not running" >&2
    sleep 20
  done
  echo "[fleet] TIMEOUT ready=$ready/$N" >&2
  exit 1
  ;;

urls)
  STAMP=$(cat "$STAMP_FILE")
  for i in $(seq 0 $((N-1))); do
    port=$((BASE_PORT+i))
    name=$(printf "sglang-omni-jaxan-%se%02d" "$STAMP" "$i")
    if curl -s -m 3 -o /dev/null "http://127.0.0.1:${port}/docs" 2>/dev/null; then
      echo "http://127.0.0.1:${port}"
    fi
  done
  ;;

down)
  # note (luojiaxuan): 只删自己前缀的容器,别人的一律不碰。
  names=$(docker ps -a --filter "name=sglang-omni-jaxan-" --format '{{.Names}}' | grep -E 'e[0-9]{2}$|p[0-9]+$' || true)
  if [ -z "$names" ]; then echo "[fleet] nothing to remove"; exit 0; fi
  echo "$names" | xargs -r docker rm -f >/dev/null 2>&1
  echo "[fleet] removed:"; echo "$names"
  left=$(docker ps -a --filter "name=sglang-omni-jaxan-" --format '{{.Names}}' | wc -l)
  echo "[fleet] remaining sglang-omni-jaxan* containers: $left"
  ;;

*)
  echo "usage: $0 {up|wait|urls|down} [N] [BASE_PORT]" >&2; exit 2 ;;
esac
