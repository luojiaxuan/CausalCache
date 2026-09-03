#!/usr/bin/env bash
# note (luojiaxuan): 贪心侦察收官后:executor 切原版 GUI-Owl-1.5-8B-Instruct(与榜单可比),
# judge 暂设零重试(key 未就位,失败即过,轨迹保留待离线补判),起 MemGUI 基线臂 A
# (官方默认历史窗 history_n=1,128 题 Pass@1,8 台后端)。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
R=/data01/jaxan/rl_v2
grep -q GREEDY_RECON_DONE $R/greedy/greedy.log || { echo "GREEDY_NOT_DONE"; grep -E "RESULT" $R/greedy/greedy.log | tail -2; exit 1; }
[ "$(pgrep -fa 'mw eval' | grep -v pgrep | wc -l)" = "0" ] || { echo "MW_EVAL_STILL_RUNNING"; exit 1; }
grep -E "RESULT|DONE" $R/greedy/greedy.log
bash /data01/jaxan/switch_executor.sh /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct
sed -i "s/^MEMGUI_LLM_RATE_LIMIT_RETRIES=.*/MEMGUI_LLM_RATE_LIMIT_RETRIES=0/; s/^MEMGUI_LLM_INFRA_RETRIES=.*/MEMGUI_LLM_INFRA_RETRIES=0/" /data01/jaxan/memgui/.env
grep -E "RETRIES" /data01/jaxan/memgui/.env | tr "\n" " "; echo
setsid bash /data01/jaxan/memgui_arm.sh armA_base_hist1 recent 1 8 > $R/memgui/armA_base_hist1.log 2>&1 < /dev/null &
sleep 40
grep -aE "Loaded|tasks|Task|Error|Traceback" $R/memgui/armA_base_hist1.log | grep -v DEBUG | tail -5 | cut -c1-160
echo "mg eval procs=$(pgrep -fa 'mg eval' | grep -v pgrep | wc -l)"
echo ARM_A_SUBMITTED
