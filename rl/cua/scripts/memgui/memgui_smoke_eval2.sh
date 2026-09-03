#!/usr/bin/env bash
# note (luojiaxuan): 按路径加载 agent 时 registry 不传 tools;改为以移植版覆盖 MemGUI 树内的
# gui_owl_1_5.py(原文件留 .bak_pre_cc),用注册名 gui_owl_1_5 跑。CC_FRAME_POLICY 未设时
# 钩子退化为官方 recency 行为(MobileWorld 上逐字节 parity 已验)。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
F=/data01/jaxan/memgui/src/mobile_world/agents/implementations/gui_owl_1_5.py
[ -f $F.bak_pre_cc ] || cp $F $F.bak_pre_cc
cp /data01/jaxan/memgui_gui_owl_cc.py $F
echo "=== registry 对路径 agent 的处理(参考) ==="; sed -n "100,134p" /data01/jaxan/memgui/src/mobile_world/agents/registry.py | grep -nE "tools|path|import|cls\(|create" | head -8 | cut -c1-140
cd /data01/jaxan/memgui
CC_FRAME_POLICY=recent CC_HISTORY_N=3 PYTHONPATH=/data01/jaxan/pyshim \
  setsid timeout 2400 uv run mg eval --agent-type gui_owl_1_5 \
  --model-name gui-owl --llm-base-url http://172.17.0.1:41041/v1 --api-key EMPTY \
  --task 001-FindProductAndFilter --max-round 50 --aw-host http://127.0.0.1:6900 \
  --log-file-root /data01/jaxan/rl_v2/memgui/smoke2 > /data01/jaxan/rl_v2/memgui/smoke2.log 2>&1 < /dev/null &
sleep 75
grep -aE "Error|error|Traceback|step|Step|action|Task" /data01/jaxan/rl_v2/memgui/smoke2.log | tail -8 | cut -c1-170
echo SMOKE2_SUBMITTED
