#!/usr/bin/env bash
# note (luojiaxuan): 宿主 inotify 实例配额(fs.inotify.max_user_instances=128,按 UID=root 全局共享)
# 被占满,只能维持 7 台 MemGUI 后端;两臂在同一组 7 台上串行补跑,runner 按前缀 mga 自动发现
# (不传 --aw-host)以便自愈。每臂前重新计算缺失任务并清空其目录。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
R=/data01/jaxan/rl_v2/memgui
docker rm -f sglang-omni-jaxan-mgb-dbg > /dev/null 2>&1; sed -i "/^sglang-omni-jaxan-mgb-dbg\t/d" "$HOME/jiaxuanluo-map.txt"
NB=$(docker ps --format "{{.Names}} {{.Status}}" | grep -E "jaxan-mga_[0-9]+ " | grep -c healthy); echo "healthy mga backends=$NB"
prep() { python3 /data01/jaxan/missing_tasks.py > /dev/null; python3 - "$1" << 'PY'
import csv, os, shutil, sys
arm = sys.argv[1]; R = "/data01/jaxan/rl_v2/memgui"
miss = [r["task_identifier"] for r in csv.DictReader(open(f"{R}/{arm}_missing.csv"))]
n = 0
for t in miss:
    d = f"{R}/{arm}/{t}"
    if os.path.isdir(d): shutil.rmtree(d); n += 1
print(f"{arm}: missing={len(miss)} cleared_dirs={n}")
PY
}
run_arm() { # arm policy hist
  prep "$1"
  MG_TASKFILE=$R/$1_missing.csv MG_PREFIX=sglang-omni-jaxan-mga MG_DISCOVER=1 bash /data01/jaxan/memgui_arm.sh "$1" "$2" "$3" "$NB" > $R/$1_rerun.log 2>&1
  echo "$(date -u +%FT%TZ) $1 rerun finished: completed=$(grep -ac 'completed on http' $R/$1_rerun.log) zero=$(grep -ac 'duration=0\.[0-9]s' $R/$1_rerun.log)"
}
run_arm armA_base_hist1 recent 1
run_arm armB_base_recency_h3 recent 3
echo SEQ_RERUN_DONE
