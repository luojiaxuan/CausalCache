#!/usr/bin/env bash
# note (luojiaxuan): curl 单流对 HF 大文件太慢(0.15GB/20min);annos.zip 已含全部标注,
# merged_anno_new.jsonl 为冗余合并版,跳过;imgs.zip 改用 huggingface_hub(xet/多线程)
# 在一次性容器里下载,支持续传。
set -uo pipefail
pkill -f "[a]ndrotmem_dl.sh"; sleep 1; pkill -f "[c]url -s -L -C - -m 7200 -o merged_anno_new.jsonl"; sleep 1
rm -f /data01/jaxan/androtmem/merged_anno_new.jsonl
docker run -d --rm --name sglang-omni-jaxan-tmp-dl --init -v /data01/jaxan:/data01/jaxan \
  -e HF_TOKEN_FILE=/data01/jaxan/token_gavinlaw -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  --entrypoint bash vllm/vllm-omni:dev -c 'export HF_TOKEN=$(cat $HF_TOKEN_FILE); cd /data01/jaxan/androtmem && python3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(\"CVC2233/AndroTMem-Bench\", \"imgs.zip\", repo_type=\"dataset\", local_dir=\"/data01/jaxan/androtmem\")
print(\"DONE\", p)
" > /data01/jaxan/androtmem/imgs_dl.log 2>&1' > /dev/null
echo "sglang-omni-jaxan-tmp-dl	gpus=none	host=$(hostname)	created=$(date -u +%FT%TZ)	desc=sglang-omni-rl 一次性下载容器(AndroTMem imgs.zip);--rm 自删" >> "$HOME/jiaxuanluo-map.txt"
sleep 45
tail -2 /data01/jaxan/androtmem/imgs_dl.log 2>/dev/null | cut -c1-160
du -sh /data01/jaxan/androtmem/.cache 2>/dev/null; ls -la /data01/jaxan/androtmem/ | grep -i imgs
echo DL2_SUBMITTED
