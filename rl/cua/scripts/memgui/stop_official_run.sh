#!/usr/bin/env bash
# note (luojiaxuan): 停止官方预算口径的两臂运行(判分 API 费用未获授权);只杀本流程的进程链。
set -uo pipefail
pkill -f "[m]emgui_official_run.sh"; pkill -f "[m]emgui_pass_loop.sh"; pkill -f "[m]emgui_arm.sh"
pkill -f "[m]g eval --agent-type gui_owl_1_5"; pkill -f "[m]emgui_rejudge.py"
sleep 3
echo "left=$(ps -eo args | grep -c '[m]emgui_official_run\|[m]emgui_pass_loop\|[m]g eval\|[m]emgui_rejudge')"
echo "$(date -u +%FT%TZ) STOPPED_BY_SPEND_HOLD" >> /data01/jaxan/rl_v2/memgui/official_run.log
