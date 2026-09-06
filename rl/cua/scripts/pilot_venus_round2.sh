#!/usr/bin/env bash
# note (luojiaxuan): B-pilot 第二轮 Venus 矩阵,复用已起好的 Venus 执行器(C/PORT 由调用方给出,容器由本线创建,按名删除):
# 等 PartMatch(Pictures 布局)的 Venus 前缀 16 题齐 → 重建规格(Mail 带请求帧;PartMatch 新前缀)→ 跑 Mail 无文本(重做)、PartMatch 原生文本 + 无文本 → 删容器与 map 行。
set -uo pipefail
P=/data01/jaxan/rl_v2/pilot; E=http://172.17.0.1:$PORT/v1; MAP=$HOME/jiaxuanluo-map.txt
until [ "$(ls $P/prefix_venus_pm2/*/result.txt 2>/dev/null | wc -l)" -ge 16 ]; do sleep 30; done
python3 /data01/jaxan/pilot_build_specs.py --prefix-dir $P/prefix_venus --seeds $P/task_seeds.txt --backend venus --out $P/specs_venus_all.jsonl --contact $P/contact_venus > $P/specs_venus.log 2>&1
grep -v '"family": "PartMatch"' $P/specs_venus_all.jsonl > $P/specs_venus_mail.jsonl
python3 /data01/jaxan/pilot_build_specs.py --prefix-dir $P/prefix_venus_pm2 --seeds $P/task_seeds.txt --backend venus --out $P/specs_venus_pm2.jsonl --contact $P/contact_venus_pm2 > $P/specs_venus_pm2.log 2>&1
echo "$(date -u +%FT%TZ) specs mail=$(wc -l < $P/specs_venus_mail.jsonl) pm2=$(wc -l < $P/specs_venus_pm2.jsonl)"; grep -v "^    " $P/specs_venus_pm2.log | head -4
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL $C"; exit 1; }
run() { rm -f $P/eval_$2.jsonl; python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $1 --backend venus --base-url $E --model UI-Venus-2 --tag $2 $3 --out $P/eval_$2.jsonl --workers 4 2>&1 | tee $P/eval_$2.txt; }
run $P/specs_venus_mail.jsonl venus_mail_notext --no-text
run $P/specs_venus_pm2.jsonl venus_pm2_text ""
run $P/specs_venus_pm2.jsonl venus_pm2_notext --no-text
docker rm -f "$C" >/dev/null 2>&1; sed -i "/^$C\t/d" "$MAP"; echo "$(date -u +%FT%TZ) removed $C"; echo PILOT_VENUS_ROUND2_DONE
