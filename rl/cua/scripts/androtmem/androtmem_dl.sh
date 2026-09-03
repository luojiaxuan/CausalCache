#!/usr/bin/env bash
# note (luojiaxuan): 下载 AndroTMem-Bench(门控已通过,gavinlaw)到持久目录,支持断点续传。
set -uo pipefail
T=$(cat /data01/jaxan/token_gavinlaw); R=CVC2233/AndroTMem-Bench; D=/data01/jaxan/androtmem
mkdir -p $D; cd $D
for f in README.md annos.zip merged_anno_new.jsonl imgs.zip; do
  echo "$(date -u +%FT%TZ) start $f"
  curl -s -L -C - -m 7200 -o "$f" "https://huggingface.co/datasets/$R/resolve/main/$f" -H "Authorization: Bearer $T" && echo "$(date -u +%FT%TZ) done $f $(stat -c %s $f) bytes"
done
echo ANDROTMEM_DL_DONE
