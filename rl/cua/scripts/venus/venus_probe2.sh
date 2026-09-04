#!/usr/bin/env bash
# note (luojiaxuan): UI-Venus-2-9B 探针 v2(外审改判版,预注册见 reviews/executor_swap_review_20260904.md §3):
# MobileWorld GUI-only 全部 117 题,四臂 = recent:0 / recent:2 / recent:8 / change2(非连续启发式 B=2),
# 程序判分、零 API 费用;池 p00–p11 并发 12,每臂 timeout 4h;逐臂汇总(全集 + heldout-39)。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
OUT=/data01/jaxan/rl_v2/venus/v2; mkdir -p $OUT
TASKS=$(python3 -c "import json;s=json.load(open('/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json'));print(','.join(sorted(s['train']+s['heldout'])))")
HOSTS=$(seq -s, -f "http://127.0.0.1:68%02g" 0 11)
cd /data01/jaxan/mw/MobileWorld
for spec in ${ARMS:-recent:0 recent:2 recent:8 change2}; do
  arm=$(echo $spec | tr -d ":")
  echo "$(date -u +%FT%TZ) START $arm"
  CC_VENUS_HIST=$spec PYTHONPATH=/data01/jaxan/pyshim timeout 14400 uv run mw eval --agent_type ui_venus2 \
    --task "$TASKS" --max_round 50 --model_name UI-Venus-2 --llm_base_url http://172.17.0.1:41041/v1 --api_key EMPTY \
    --step_wait_time 3 --max-concurrency 12 --aw-host "$HOSTS" --log_file_root $OUT/$arm > $OUT/$arm.log 2>&1
  echo "$(date -u +%FT%TZ) DONE $arm rc=$?"
  python3 /data01/jaxan/venus_tally2.py $OUT recent0 recent2 recent8 change2 2>/dev/null
done
echo VENUS_PROBE2_DONE
