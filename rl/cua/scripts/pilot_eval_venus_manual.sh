#!/usr/bin/env bash
# note (luojiaxuan): 对一台已经起好的 Venus 执行器(C/PORT/OWN 由调用方给出,OWN = 该容器 cc.owner 标签)跑 Mail 两族的干预矩阵,
# 原生文本与无文本各一遍,跑完删容器。用于自动链被中止后接管其容器,避免重新装载模型。
set -uo pipefail
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot; E=http://172.17.0.1:$PORT/v1
for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) venus ready ($C:$PORT) specs=$(wc -l < $P/specs_venus_mail.jsonl)"
for proto in text notext; do
  extra=$([ "$proto" = notext ] && echo --no-text || true)
  rm -f $P/eval_venus_mail_$proto.jsonl
  python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/specs_venus_mail.jsonl --backend venus --base-url $E --model UI-Venus-2 \
    --tag venus_mail_$proto $extra --out $P/eval_venus_mail_$proto.jsonl --workers 4 2>&1 | tee $P/eval_venus_mail_$proto.txt
done
rm_own "$C"; echo PILOT_EVAL_VENUS_MAIL_DONE
