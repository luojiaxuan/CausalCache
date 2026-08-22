#!/bin/bash
# note (luojiaxuan): AndroidWorld 基线复现 v2 —— hyper01 准备脚本(GPU7 policy 容器 +
# OCR 重物化探针 + 官方 seed30 roster 生成)。每步带 STEP 标记,失败不静默。
set -u
cd /data04/jaxan

# STEP 1: policy 容器(--network host 为连宿主 emulator 端口;仅暴露 GPU7)
docker rm -f sglang-omni-jaxan-1 >/dev/null 2>&1 || true
docker run -d --name sglang-omni-jaxan-1 --gpus '"device=7"' --network host \
  --ipc=host --shm-size=32g -v /data04/jaxan:/data \
  -v /data04/jaxan/cache/huggingface:/root/.cache/huggingface \
  hongccc/sglang-omni:dev sleep infinity >/dev/null && echo STEP_CONTAINER_OK
printf "sglang-omni-jaxan-1\tgpus=7\thost=hyper01\tcreated=%s\tdesc=AndroidWorld 基线复现 v2 policy;收尾:116 局评完即删\n" \
  "$(date -u +%FT%TZ)" >> $HOME/jiaxuanluo-map.txt

# STEP 2: 容器内依赖(transformers 钉 5.6.0 与 OSWorld 侧一致;rapidocr 按仓库 pinned 3.8.4)
docker exec sglang-omni-jaxan-1 bash -lc '
  pip install -q "transformers==5.6.0" "rapidocr==3.8.4" "onnxruntime" 2>&1 | tail -1
  mkdir -p /root/tmp /data/awfleet
  echo STEP_DEPS_OK'

# STEP 3: OCR 探针 —— 让 PinnedOnlineOCRProvider 自己报缺什么文件
docker exec sglang-omni-jaxan-1 bash -lc '
  cd /data/osworld/CausalCache && mkdir -p /data/awfleet/ocr_models
  PYTHONPATH=code:code/scripts python3 - << "PYEOF" 2>&1 | tail -15
from pathlib import Path
from causalcache.set_utility_androidworld import PinnedOnlineOCRProvider
try:
    p = PinnedOnlineOCRProvider.load(
        backend_config_path=Path("code/configs/restoration_v2_ocr_backend.json"),
        backend_manifest_path=Path("data/manifests/restoration_v2_ocr_backend.json"),
        model_dir=Path("/data/awfleet/ocr_models"))
    print("OCR_PROBE_LOADED_OK")
except Exception as e:
    print("OCR_PROBE_ERR:", type(e).__name__, str(e)[:600])
PYEOF
  echo STEP_OCR_PROBE_DONE'

# STEP 4: 官方 seed30 roster(离线生成;先在 policy 容器试,失败则记录待 env 容器)
docker exec sglang-omni-jaxan-1 bash -lc '
  cd /data/osworld/CausalCache
  PYTHONPATH=code:code/scripts python3 code/scripts/build_androidworld_official_seed_plan.py --help 2>&1 | head -12
  echo STEP_PLAN_HELP_DONE'

echo AW_SETUP_H01_DONE
