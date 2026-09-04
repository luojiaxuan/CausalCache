#!/usr/bin/env bash
# note (luojiaxuan): MemGUI 补跑的多轮收敛循环。runner 自带的"重建不健康后端"会无限循环
# (重建出的容器无 init、常常起不健康),病后端以 0 秒假完成吞掉任务。改为每轮:
#  (1) 我方接管后端健康:删掉不健康/无 init 的 mga_*,按缺失编号带 --init 重起,等 healthy;
#  (2) 计算该臂轨迹为空的任务 → 只跑这些(按前缀自动发现,并发=健康后端数);
#  (3) 收敛判据:无缺失任务;最多 PASSES 轮。后端数压到 5 台,给宿主 inotify 配额留余量。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
R=/data01/jaxan/rl_v2/memgui; NBACK=${NBACK:-5}; PASSES=${PASSES:-4}
IMG=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep memgui-bench | head -1)
launch() { docker run -d --rm --init --privileged --ulimit nofile=524288:524288 --name "sglang-omni-jaxan-mga_$1" \
    -p $((6900+$1)):6800 -p $((7900+$1)):7860 -p $((5700+$1)):5556 -v /data01/jaxan/memgui/.env:/app/service/.env -e EMULATOR_TIMEOUT=1200 \
    --entrypoint /usr/local/bin/memgui-runtime-entrypoint.sh "$IMG" tail -f /var/log/emulator.log /var/log/server.log /var/log/dockerd.err.log > /dev/null 2>&1; }
ensure_backends() {
  for c in $(docker ps -a --format "{{.Names}}" | grep -E "^sglang-omni-jaxan-mga_[0-9]+$"); do
    idx=${c##*_}; st=$(docker inspect -f "{{.State.Health.Status}} {{.HostConfig.Init}}" $c 2>/dev/null)
    if [ $idx -ge $NBACK ] || ! echo "$st" | grep -q "healthy true"; then docker rm -f $c > /dev/null 2>&1; echo "  removed $c ($st)"; fi
  done
  for i in $(seq 0 $((NBACK-1))); do docker ps --format "{{.Names}}" | grep -q "^sglang-omni-jaxan-mga_$i$" || { launch $i; echo "  launched mga_$i"; sleep 20; }; done
  for t in $(seq 1 40); do h=$(docker ps --format "{{.Names}} {{.Status}}" | grep -E "jaxan-mga_" | grep -c healthy); [ "$h" -ge "$NBACK" ] && break; sleep 15; done
  sed -i "/^sglang-omni-jaxan-mga_/d" "$HOME/jiaxuanluo-map.txt"
  for c in $(docker ps --format "{{.Names}}" | grep "jaxan-mga_"); do echo "$c	gpus=none	host=$(hostname)	created=$(date -u +%FT%TZ)	desc=sglang-omni-rl MemGUI-Bench 后端模拟器(--init);调用方=本机 mg eval;保留至 2026-09-06 PT ⚠ 在用勿删;收尾:评测线结束删" >> "$HOME/jiaxuanluo-map.txt"; done
  echo "$(date -u +%FT%TZ) healthy backends=$h/$NBACK"
}
missing_csv() { python3 /data01/jaxan/missing_tasks.py > /dev/null; python3 - "$1" << 'PY'
import csv, os, shutil, sys, glob
arm = sys.argv[1]; R = "/data01/jaxan/rl_v2/memgui"
miss = [r["task_identifier"] for r in csv.DictReader(open(f"{R}/{arm}_missing.csv"))]
for t in miss:
    for d in glob.glob(f"{R}/{arm}/{t}*"):
        if os.path.isdir(d): shutil.rmtree(d)
print(len(miss))
PY
}
for pass_i in $(seq 1 $PASSES); do
  echo "=== PASS $pass_i $(date -u +%FT%TZ) ==="; ensure_backends
  NB=$(docker ps --format "{{.Names}} {{.Status}}" | grep -E "jaxan-mga_" | grep -c healthy); [ "$NB" -lt 1 ] && { echo "NO_HEALTHY_BACKEND"; sleep 300; continue; }
  left=0
  for spec in "armB_base_recency_h3 recent 3" "armA_base_hist1 recent 1"; do set -- $spec; arm=$1
    m=$(missing_csv $arm); echo "  $arm missing=$m"; [ "$m" = "0" ] && continue; left=$((left+m))
    MG_TASKFILE=$R/${arm}_missing.csv MG_PREFIX=sglang-omni-jaxan-mga MG_DISCOVER=1 bash /data01/jaxan/memgui_arm.sh $arm $2 $3 $NB > $R/${arm}_pass$pass_i.log 2>&1
    echo "  $arm pass$pass_i: completed=$(grep -ac 'completed on http' $R/${arm}_pass$pass_i.log) zero=$(grep -ac 'duration=0\.[0-9]s' $R/${arm}_pass$pass_i.log)"
  done
  [ "$left" = "0" ] && { echo "ALL_TASKS_HAVE_TRAJ"; break; }
done
python3 /data01/jaxan/missing_tasks.py
echo PASS_LOOP_DONE
