#!/bin/bash
# note (luojiaxuan): 基线复现 v2 发射序列(hyper01 宿主侧,nohup 后台)。
# 双修:一次性 emulator(每局一台,116 台)+ vLLM 服务栈。STEP 门逐级放行。
set -u
cd /data04/jaxan/awfleet
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# STEP V: vLLM server(GPU7,host 网络)
if ! curl -s -m 3 http://127.0.0.1:41999/v1/models | grep -q gui-owl; then
  docker rm -f sglang-omni-jaxan-1v >/dev/null 2>&1 || true
  docker run -d --name sglang-omni-jaxan-1v --gpus '"device=7"' --network host \
    --ipc=host --shm-size=16g -v /data04/jaxan:/data \
    vllm/vllm-openai:v0.27.0 \
    --model /data/models/GUI-Owl-1.5-8B-Instruct --served-model-name gui-owl \
    --port 41999 --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image": 8}' >/dev/null
  printf "sglang-omni-jaxan-1v\tgpus=7\thost=hyper01\tcreated=%s\tdesc=AW基线v2 vLLM server;与 jaxan-1 同任务;收尾:评完即删\n" "$(date -u +%FT%TZ)" >> $HOME/jiaxuanluo-map.txt
fi
for i in $(seq 1 60); do
  curl -s -m 3 http://127.0.0.1:41999/v1/models | grep -q gui-owl && break; sleep 10
done
curl -s -m 3 http://127.0.0.1:41999/v1/models | grep -q gui-owl && LOG STEP_VLLM_READY || { LOG STEP_VLLM_FAIL; docker logs --tail 15 sglang-omni-jaxan-1v; exit 1; }

# STEP R: roster(seed30 × 1 comb;先 policy 容器,失败退 env 镜像)
if [ ! -s seed30_plan.json ]; then
  docker exec sglang-omni-jaxan-1 bash -lc 'cd /data/osworld/CausalCache && PYTHONPATH=code:code/scripts python3 code/scripts/build_androidworld_official_seed_plan.py --templates-from /data/awfleet/templates116.txt --output /data/awfleet/seed30_plan.json --seed 30 --combinations 1' 2>roster.err \
  || docker run --rm -v /data04/jaxan:/data -v /data04/jaxan/osworld/CausalCache:/repo jaxanluo/sglang-omni:env bash -lc 'cd /repo && PYTHONPATH=/:code:code/scripts python3 code/scripts/build_androidworld_official_seed_plan.py --templates-from /data/awfleet/templates116.txt --output /data/awfleet/seed30_plan.json --seed 30 --combinations 1' 2>>roster.err
fi
[ -s seed30_plan.json ] && LOG STEP_ROSTER_OK || { LOG STEP_ROSTER_FAIL; tail -5 roster.err; exit 1; }
python3 - << 'PYEOF'
import json
plan=json.load(open('/data04/jaxan/awfleet/seed30_plan.json'))
inst=plan['instances'] if isinstance(plan,dict) and 'instances' in plan else plan
out=[{"task_type":r["task_type"],"task_index":int(r.get("task_index",0))} for r in inst]
json.dump(out,open('/data04/jaxan/awfleet/plan_instances116.json','w'))
print("PLAN_INSTANCES",len(out))
PYEOF

# STEP F: 116 台一次性 emulator,4 批 × 29,批间 120s
for s in 0 29 58 87; do
  bash /data04/jaxan/awfleet/aw_fleet.sh up 29 42001 $s
  sleep 120
done
bash /data04/jaxan/awfleet/aw_fleet.sh wait 116 42001 && LOG STEP_FLEET_READY || LOG STEP_FLEET_PARTIAL
bash /data04/jaxan/awfleet/aw_fleet.sh urls 116 42001 > ready_urls.txt
LOG "urls=$(wc -l < ready_urls.txt)"

# STEP S: 单局 smoke(首个 url,1 实例)
head -1 ready_urls.txt > smoke_url.txt
python3 -c "import json;json.dump([json.load(open('/data04/jaxan/awfleet/plan_instances116.json'))[0]],open('/data04/jaxan/awfleet/smoke1.json','w'))"
docker exec sglang-omni-jaxan-1 bash -lc "cd /data/osworld/CausalCache && PYTHONPATH=code:code/scripts python3 code/scripts/run_official_androidworld.py --repository-root . --model-dir /data/models/GUI-Owl-1.5-8B-Instruct --ocr-model-dir /data/awfleet/ocr_models --ceiling-plan /data/awfleet/seed30_plan.json --plan-instances /data/awfleet/smoke1.json --output-root /data/awfleet/smoke_out --device vllm --vllm-endpoint http://127.0.0.1:41999 --base-url $(cat smoke_url.txt)" > smoke_run.log 2>&1
grep -q '"complete"' smoke_run.log && LOG STEP_SMOKE_OK || { LOG STEP_SMOKE_FAIL; tail -15 smoke_run.log; exit 1; }

# STEP M: 正式 116 局(全部 ready url,一 url 一线程一局;skip-existing 续跑)
URLS=$(awk '{printf " --base-url %s", $0}' ready_urls.txt)
docker exec sglang-omni-jaxan-1 bash -lc "cd /data/osworld/CausalCache && PYTHONPATH=code:code/scripts nohup python3 code/scripts/run_official_androidworld.py --repository-root . --model-dir /data/models/GUI-Owl-1.5-8B-Instruct --ocr-model-dir /data/awfleet/ocr_models --ceiling-plan /data/awfleet/seed30_plan.json --plan-instances /data/awfleet/plan_instances116.json --output-root /data/awfleet/v2_seed30 --device vllm --vllm-endpoint http://127.0.0.1:41999 $URLS > /data/awfleet/v2_run.log 2>&1 &"
LOG STEP_MAIN_FIRED
echo AW_LAUNCH_V2_DONE
