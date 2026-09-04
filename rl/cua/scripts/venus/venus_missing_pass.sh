#!/usr/bin/env bash
# note (luojiaxuan): 探针 v2 补跑:runner 层报错(模拟器截图超时等)的任务没有 result.txt,按臂只重跑这些;
# 等离线 G2 收官后执行(共用 GPU1),最多 2 轮,之后终表。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
O=/data01/jaxan/rl_v2/venus/v2
while ! grep -q G2_DONE /data01/jaxan/rl_v2/venus/g2_waiter.log 2>/dev/null; do sleep 120; done
HOSTS=$(seq -s, -f "http://127.0.0.1:68%02g" 0 11)
cd /data01/jaxan/mw/MobileWorld
for pass in 1 2; do
  for spec in recent:0 recent:2 recent:8 change2; do
    arm=$(echo $spec | tr -d ":")
    MISSING=$(python3 - $O/$arm <<'PY'
import json, os, sys
root = sys.argv[1]; s = json.load(open("/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json"))
print(",".join(t for t in sorted(s["train"] + s["heldout"]) if not os.path.exists(os.path.join(root, t, "result.txt"))))
PY
)
    [ -z "$MISSING" ] && { echo "$arm: none missing"; continue; }
    echo "$(date -u +%FT%TZ) pass$pass $arm missing: $MISSING"
    for t in ${MISSING//,/ }; do rm -rf "$O/$arm/$t"; done
    CC_VENUS_HIST=$spec PYTHONPATH=/data01/jaxan/pyshim timeout 7200 uv run mw eval --agent_type ui_venus2 \
      --task "$MISSING" --max_round 50 --model_name UI-Venus-2 --llm_base_url http://172.17.0.1:41041/v1 --api_key EMPTY \
      --step_wait_time 3 --max-concurrency 12 --aw-host "$HOSTS" --log_file_root $O/$arm >> $O/${arm}_missing.log 2>&1
  done
done
python3 /data01/jaxan/venus_tally2.py $O recent0 recent2 recent8 change2 | tee $O/final_tally.txt
echo MISSING_PASS_DONE
