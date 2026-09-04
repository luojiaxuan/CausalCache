#!/usr/bin/env bash
# note (luojiaxuan): 温度对照臂。四臂探针用官方离线示例脚本的默认 temperature=0.0,而模型卡对 agent 类任务
# 建议 1.0;我们的 recency-2 只有 45.7% 而论文报 65.8%,温度是首要嫌疑。同 117 题、同协议,只改温度。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
O=/data01/jaxan/rl_v2/venus/v2; mkdir -p $O
TASKS=$(python3 -c "import json;s=json.load(open('/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json'));print(','.join(sorted(s['train']+s['heldout'])))")
HOSTS=$(seq -s, -f "http://127.0.0.1:68%02g" 0 11)
cd /data01/jaxan/mw/MobileWorld
echo "$(date -u +%FT%TZ) START recent2_t1"
CC_VENUS_HIST=recent:2 CC_VENUS_TEMP=1.0 PYTHONPATH=/data01/jaxan/pyshim timeout 14400 uv run mw eval --agent_type ui_venus2 \
  --task "$TASKS" --max_round 50 --model_name UI-Venus-2 --llm_base_url http://172.17.0.1:41041/v1 --api_key EMPTY \
  --step_wait_time 3 --max-concurrency 12 --aw-host "$HOSTS" --log_file_root $O/recent2_t1 > $O/recent2_t1.log 2>&1
echo "$(date -u +%FT%TZ) DONE recent2_t1"
python3 /data01/jaxan/venus_tally2.py $O recent2 recent2_t1
echo TEMP_ARM_DONE
