#!/bin/bash
# note (luojiaxuan): held-out 方向读评测(双臂配对,方向 only,不算显著性)。
#   臂 A rl    :rl serve(τ=0 部署 argmax)+ iter-$ITER 的 selector 头 + HGKV LoRA
#   臂 B recent:主仓库官方 serve(冻结策略,无 selector 无 LoRA),worker 走
#               --memory-arm recent —— 论文时代的部署默认基线
# 协议:held-out 120 任务 × 每臂 1 轮 × 50 步 × B=2。断点续跑(result.json 跳过);
# 每臂结束做一次简易修复遍(策略失败>0 → 重启 server 重跑,续跑令已完成秒过)。
# 前置:训练 loop 必须已停(共用 GPU 与 worker 位);跑完由调用方重启 loop。
# 用法(h00):ITER=12 nohup rl/runscripts/rl_heldout_eval.sh > eval.log 2>&1 &
set -u
ITER=${ITER:?需要 ITER=权重迭代号}
GPUS=${GPUS:-"3 4 5"}; GPU_ARR=($GPUS); SERVERS=${SERVERS:-${#GPU_ARR[@]}}
WORKERS=${WORKERS:-8}; MAX_STEPS=${MAX_STEPS:-50}; BUDGET=${BUDGET:-2}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-300}

CTN=${CTN:-sglang-omni-jaxan}
B=${B:-/data04/jaxan/osworld}
RLH=${RLH:-/data04/jaxan/rl}
RLC=${RLC:-/data/rl}
REPO=${REPO:-/data/osworld/CausalCache}
WREPO=${WREPO:-/data04/jaxan/osworld/CausalCache}
CACHE=${CACHE:-$B/cache-fast}
MODEL=${MODEL:-/data/models/GUI-Owl-1.5-8B-Instruct}
# 轮次隔离(正式评测 2 轮可并行):PORT0 与 TAG 各占一套端口与输出目录,
# 配 GPUS 分组即可同机双轮,墙钟减半
PORT0=${PORT0:-19511}
TAG=${TAG:-}
META=rl_heldout_v1.json

EV_H=$RLH/eval-iter$ITER$TAG; EV_C=$RLC/eval-iter$ITER$TAG
mkdir -p "$EV_H"
log() { echo "[$(date -u +%FT%TZ)] eval-iter$ITER $*"; }
cexec() { docker exec "$CTN" bash -lc "$*"; }

CIP=$(docker inspect "$CTN" --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
case "$CIP" in ""|*[!0-9.]*) CIP=127.0.0.1;; esac

# held-out meta 从仓库清单放入 OSWorld(worker 以 osworld-root 相对路径读)
[ -f "$B/OSWorld/evaluation_examples/$META" ] \
  || cp "$WREPO/rl/data/manifests/osworld_rl_heldout_v1.json" \
        "$B/OSWorld/evaluation_examples/$META"

# 杀 server 并等进程真正消失(eval12 实测教训:pkill 后 CUDA 拆卸要 5-20s,
# 垂死进程仍应答 /health 200,骗过下一臂健康检查 → 整臂 35 秒全灭收 0 条)
# 按本轮端口精确狙击:并行双轮时全局 pkill 会误杀另一轮的 server
kill_servers_wait() {
  local s i alive
  for s in $(seq 0 $((SERVERS-1))); do
    cexec "pkill -f '[s]erve_osworld.*--port $((PORT0+s))' || true"
  done
  for i in $(seq 1 20); do
    alive=0
    for s in $(seq 0 $((SERVERS-1))); do
      cexec "pgrep -f '[s]erve_osworld.*--port $((PORT0+s))' >/dev/null" && alive=1
    done
    [ "$alive" = "0" ] && return 0
    sleep 3
  done
  log "WARN server 30s 内未退净,继续(端口绑定由新 server 报错兜底)"
}

launch_servers() {  # $1 = rl | recent
  if [ "$1" = "rl" ]; then
    SRV="rl/code/scripts/serve_osworld_rl_policy.py"
    SHA=$(cexec "sha256sum $RLC/iter-$ITER/adapter.pt | cut -d' ' -f1")
    EXTRA="--selector-mode hidden --selector-head $RLC/iter-$ITER/selector_bundle.pt \
      --adapter-checkpoint $RLC/iter-$ITER/adapter.pt --adapter-checkpoint-sha256 $SHA"
  else
    SRV="code/scripts/serve_osworld_official_policy.py"
    EXTRA=""
  fi
  for s in $(seq 0 $((SERVERS-1))); do
    docker exec -d "$CTN" bash -lc "cd $REPO && CUDA_VISIBLE_DEVICES=${GPU_ARR[$s]} \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=code \
      python3 $SRV \
      --model-dir $MODEL --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
      --device cuda:0 --port $((PORT0+s)) --visual-tokens 2560 \
      --memory-budget $BUDGET $EXTRA >> $EV_C/serve-$1-$s.log 2>&1"
  done
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
  [ "$ok" = "1" ] || { log "FATAL 臂 $1 server 健康窗口耗尽"; exit 1; }
}

run_arm_workers() {  # $1 = rl | recent
  # 两臂 worker 都传 --memory-arm full(worker 只有 summary|full 两选项;
  # recent-B 是 server 侧行为:官方 serve 不带 selector 即默认 recent-B 填充,
  # profile id "..._osworld_recent_b4_v1"。eval12 两轮全灭的真凶就是这里
  # 曾误传 'recent' → 8 worker argparse 秒退收 0)
  local memarm=full
  for s in $(seq 0 $((WORKERS-1))); do
    ( cd "$B/run30" && PYTHONPATH=$WREPO/code:$B/OSWorld /usr/bin/python3 \
        "$WREPO/code/scripts/run_osworld_benchmark_worker.py" \
        --osworld-root "$B/OSWorld" \
        --meta-path "evaluation_examples/$META" \
        --shard-index "$s" --shard-count "$WORKERS" \
        --output-root "$EV_H/out-$1" \
        --policy-endpoint "http://$CIP:$((PORT0 + s % SERVERS))/act" \
        --memory-arm "$memarm" --memory-budget "$BUDGET" \
        --max-steps "$MAX_STEPS" \
        --cache-dir "$CACHE" \
        >> "$EV_H/worker-$1-$s.log" 2>&1 ) &
  done
  wait
}

for arm in rl recent; do
  if [ ! -f "$EV_H/ARM_${arm}_DONE" ]; then
    cexec "mkdir -p $EV_C/out-$arm && chmod -R 777 $EV_C" || mkdir -p "$EV_H/out-$arm"
    log "臂 $arm:启动 $SERVERS server + $WORKERS worker(τ=0,$MAX_STEPS 步)"
    launch_servers "$arm"
    run_arm_workers "$arm"
    PF=$(cexec "cat $EV_C/serve-$arm-*.log 2>/dev/null | grep -c OSWORLD_POLICY_FAILURE" | tr -cd '0-9')
    if [ "${PF:-0}" -gt 0 ]; then
      log "臂 $arm:$PF 次策略失败,重启补跑一遍"
      kill_servers_wait
      launch_servers "$arm"
      run_arm_workers "$arm"
    fi
    kill_servers_wait
    n=$(find "$EV_H/out-$arm" -name result.json 2>/dev/null | wc -l)
    # 收 0 条 = 系统性故障(如 server 换臂竞态),绝不盖章
    if [ "$n" = "0" ]; then log "FATAL 臂 $arm 收 0 条"; exit 1; fi
    log "臂 $arm:收 $n 条,server 已回收"
    touch "$EV_H/ARM_${arm}_DONE"
  fi
done

# ---- 配对汇总(方向 only)----
python3 - "$EV_H" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
def arm_results(name):
    # 布局:out-<arm>/<domain>/<task_id>/result.json,成败字段 = success(bool)
    out = {}
    for f in (root / f"out-{name}").rglob("result.json"):
        try:
            out[f.parent.name] = int(bool(json.loads(f.read_text()).get("success")))
        except Exception:
            pass
    return out
rl, rec = arm_results("rl"), arm_results("recent")
common = sorted(set(rl) & set(rec))
win = [t for t in common if rl[t] > rec[t]]
lose = [t for t in common if rl[t] < rec[t]]
summary = {
    "paired_tasks": len(common),
    "rl_success": sum(rl[t] for t in common),
    "recent_success": sum(rec[t] for t in common),
    "rl_wins": len(win), "rl_losses": len(lose),
    "win_tasks": win[:20], "loss_tasks": lose[:20],
    "rl_only_count": len(rl), "recent_only_count": len(rec),
}
(root / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary))
PY
log "EVAL_DONE"
