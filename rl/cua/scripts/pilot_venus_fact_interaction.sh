#!/usr/bin/env bash
# note (luojiaxuan): 外审点名的交互对照:同长度同风格的一行历史(只留 action),证据轮的事实"留"(Noted: <事实>)vs "删",交叉 源帧轮 / 同龄对照轮 / 不给图;
# 估计 [(src−ctrl)_删 − (src−ctrl)_留]。Venus,Mail 30 checkpoint;跑完删。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41201 41211 41221 41231 41241 41271)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim -p 172.17.0.1:$PORT:8000 \
  vllm/vllm-omni:dev --model /data01/jaxan/models/UI-Venus-2-9b --served-model-name UI-Venus-2 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl B-pilot 事实留/删 × 源帧/对照 交互对照(Venus,Mail 30);⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) venus up ($C GPU$G:$PORT)"
for mode in action_fact action; do rm -f $P/eval_venus_mail_fx_$mode.jsonl
  python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/specs_venus_mail.jsonl --backend venus --base-url $E --model UI-Venus-2 --tag venus_mail_fx_$mode --text-mode $mode --conds text_only,ctrl_at_turn,src_at_turn,gold_text --out $P/eval_venus_mail_fx_$mode.jsonl --workers 4 2>&1 | tee $P/eval_venus_mail_fx_$mode.txt; done
python3 - <<'PY'
import json, random
P="/data01/jaxan/rl_v2/pilot"
def load(m): return {r["task"]: r for r in (json.loads(l) for l in open(f"{P}/eval_venus_mail_fx_{m}.jsonl"))}
om, re_ = load("action"), load("action_fact"); tasks = sorted(set(om) & set(re_))
def d(rows, t): return rows[t]["hit"]["src_at_turn"] - rows[t]["hit"]["ctrl_at_turn"]
pairs = {}
for t in tasks: pairs.setdefault(om[t]["pair"], []).append(t)
def stat(ts): return 100 * sum(d(om, t) - d(re_, t) for t in ts) / len(ts)
point = stat(tasks); keys = list(pairs); rng = random.Random(0); bs = []
for _ in range(4000):
    ts = [t for k in rng.choices(keys, k=len(keys)) for t in pairs[k]]; bs.append(stat(ts))
bs.sort(); print(f"interaction (src-ctrl)_omitted - (src-ctrl)_retained = {point:+.1f} pp  95% CI [{bs[100]:+.1f}, {bs[3899]:+.1f}]  n={len(tasks)} pairs={len(keys)}")
for m, rows in (("omitted", om), ("retained", re_)):
    print(m, {c: round(sum(r["hit"][c] for r in rows.values())/len(rows), 3) for c in ("text_only", "ctrl_at_turn", "src_at_turn", "gold_text")}, "leak", sum(r["leak"]["hist"] for r in rows.values()))
PY
rm_own "$C"; echo PILOT_FACT_INTERACTION_DONE
