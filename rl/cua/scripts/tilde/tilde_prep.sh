#!/usr/bin/env bash
# note (luojiaxuan): Tilde 登录节点侧准备(不占 GPU):建 vLLM venv、拉 GUI-Owl 权重与轨迹分片并解包。
# 计算节点是否有外网未知,所以所有下载都在登录节点做完,作业里只读 $HOME。
set -uo pipefail
R=$HOME/jaxan/runs/guiowl_label
mkdir -p $R/data $R/logs
cd $R
export HF_TOKEN=$(cat $HOME/.cache/huggingface/token_gavinlaw)
if [ ! -x $R/venv/bin/python ]; then
  uv venv --python 3.12 $R/venv && $R/venv/bin/python -m ensurepip 2>/dev/null
  VIRTUAL_ENV=$R/venv uv pip install --quiet vllm huggingface_hub
fi
$R/venv/bin/python - <<'PY'
import os, tarfile, glob
from huggingface_hub import snapshot_download
R = os.path.expanduser("~/jaxan/runs/guiowl_label")
m = snapshot_download("mPLUG/GUI-Owl-1.5-8B-Instruct", local_dir=f"{R}/GUI-Owl-1.5-8B-Instruct", max_workers=8)
print("model:", m, flush=True)
d = snapshot_download("gavinlaw/causalcache-mw-guiowl-success-trajs", repo_type="dataset",
                      local_dir=f"{R}/packs", max_workers=8)
print("packs:", d, flush=True)
os.makedirs(f"{R}/data", exist_ok=True)
for t in sorted(glob.glob(f"{R}/packs/*.tar")):
    with tarfile.open(t) as tf:
        tf.extractall(f"{R}/data")
    print("extracted", os.path.basename(t), flush=True)
print("trajs:", len(glob.glob(f"{R}/data/traj_*/*/result.txt")), flush=True)
PY
df -h $HOME | tail -1
echo TILDE_PREP_DONE
