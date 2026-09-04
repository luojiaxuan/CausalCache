#!/usr/bin/env bash
# note (luojiaxuan): 用户指令:AndroTMem 离线数据集线全部停止并删除。正本为公开 HF 数据集,可重取。
set -uo pipefail
pkill -f "[a]ndrotmem_probe" ; pkill -f "[p]robe_actions_chain" ; pkill -f "[p]robe_fair.sh"; pkill -f "[p]robe32_chain"; sleep 2
for c in sglang-omni-jaxan-rle59 sglang-omni-jaxan-rle32; do docker rm -f -v $c > /dev/null 2>&1 && echo "removed $c"; sed -i "/^$c\t/d" "$HOME/jiaxuanluo-map.txt"; done
for d in /data01/jaxan/androtmem /data01/jaxan/AndroTMem /data04/jaxan/androtmem /data04/jaxan/AndroTMem; do [ -d $d ] && { rm -rf $d; echo "deleted $d"; }; done
echo "probe procs=$(pgrep -f '[a]ndrotmem_probe' | wc -l)"
echo "containers=$(docker ps -a --format '{{.Names}}' | grep -c '^sglang-omni-jaxan') map=$(grep -c '^sglang-omni-jaxan' $HOME/jiaxuanluo-map.txt)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -3
