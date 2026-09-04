#!/usr/bin/env bash
# note (luojiaxuan): 停止 Venus 余量探针进程链(重发修订版前);只杀本流程。
pkill -f "[v]enus_probe.sh"; pkill -f "[m]w eval --agent_type ui_venus2"; sleep 3
echo "left=$(ps -eo args | grep -c '[v]enus_probe\|[m]w eval --agent_type ui_venus2')"
