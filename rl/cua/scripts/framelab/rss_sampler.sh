#!/usr/bin/env bash
# note (luojiaxuan): selector 服务宿主 RSS 每分钟一采,用于估计泄漏斜率与重启周期。
while true; do
  echo "$(date -u +%FT%TZ) $(docker stats --no-stream --format '{{.MemUsage}}' sglang-omni-jaxan-rls 2>/dev/null)" >> /data01/jaxan/rl_v2/rls_rss.log
  sleep 60
done
