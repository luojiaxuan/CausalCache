#!/usr/bin/env bash
# note (luojiaxuan): 方向 B(更强的冻结 executor)侦察的准备:在 hyper01 后台下载 GUI-Owl-1.5-32B 权重
# 与 AndroTMem-Bench 数据(annos.zip + imgs.zip),不占 GPU;用一次性容器跑 huggingface_hub。
set -uo pipefail
mkdir -p /data04/jaxan/models /data04/jaxan/androtmem
docker run -d --rm --name sglang-omni-jaxan-tmp-dl --init -v /data04/jaxan:/data04/jaxan -e HF_TOKEN_FILE=/data04/jaxan/awfleet/token_gavinlaw --entrypoint bash vllm/vllm-omni:v0.28.0rc1 -c 'export HF_TOKEN=$(cat $HF_TOKEN_FILE); python3 - << "PY" > /data04/jaxan/androtmem/prep_b.log 2>&1
from huggingface_hub import snapshot_download, hf_hub_download
for f in ("annos.zip", "imgs.zip"):
    print("dl", f, hf_hub_download("CVC2233/AndroTMem-Bench", f, repo_type="dataset", local_dir="/data04/jaxan/androtmem"), flush=True)
p = snapshot_download("mPLUG/GUI-Owl-1.5-32B-Instruct", local_dir="/data04/jaxan/models/GUI-Owl-1.5-32B-Instruct")
print("MODEL_DONE", p, flush=True)
PY
echo PREP_B_DONE >> /data04/jaxan/androtmem/prep_b.log' > /dev/null
echo "sglang-omni-jaxan-tmp-dl	gpus=none	host=$(hostname)	created=$(date -u +%FT%TZ)	desc=sglang-omni-rl 一次性下载容器(GUI-Owl-1.5-32B + AndroTMem 数据);--rm 自删" >> "$HOME/jiaxuanluo-map.txt"
sleep 20; tail -2 /data04/jaxan/androtmem/prep_b.log 2>/dev/null | cut -c1-120; echo PREP_B_SUBMITTED
