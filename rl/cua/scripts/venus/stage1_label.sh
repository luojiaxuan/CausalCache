#!/usr/bin/env bash
# note (luojiaxuan): Stage I 标注(动态 B):在 Venus 自身成功轨迹的状态上,**保留其文本推理历史**(部署条件),
# 只改历史截图集合,枚举 23 个上下文 = B0(空)+ 6 单帧 + 15 帧对 + recency 基线,贪心解码后与该轨迹自身的
# 参考动作做等价类匹配。产出即 Stage I 的标签集:任一策略(含 B=0 门控)都能在其上**离线精确评估**,
# 无需再跑 rollout。两台 Venus 服务并行(GPU1/GPU3),按 arm 切分轨迹避免写同一文件。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
O=/data01/jaxan/rl_v2/venus; V=$O/v2; cd /data01/jaxan/mw/MobileWorld
until curl -s -m 5 http://172.17.0.1:41042/v1/models > /dev/null; do sleep 20; done
uv run python /data01/jaxan/venus_oracle.py --base-url http://172.17.0.1:41041/v1 \
  --roots $V/recent0 $V/recent2 --out $O/stage1_labels_a.jsonl --per-traj 12 --limit 1400 --workers 12 \
  > $O/stage1_label_a.log 2>&1 &
uv run python /data01/jaxan/venus_oracle.py --base-url http://172.17.0.1:41042/v1 \
  --roots $V/recent8 $V/change2 --out $O/stage1_labels_b.jsonl --per-traj 12 --limit 1400 --workers 12 \
  > $O/stage1_label_b.log 2>&1 &
wait
python3 /data01/jaxan/venus_oracle_verdict.py $O/stage1_labels_a.jsonl $O/stage1_labels_b.jsonl | tee $O/stage1_labels_verdict.txt
echo STAGE1_LABEL_DONE
