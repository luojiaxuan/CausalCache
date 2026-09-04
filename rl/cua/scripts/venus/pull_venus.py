# note (luojiaxuan): 拉取 UI-Venus-2-9b 权重到 hyper00 单一根目录下的 models/;公开仓不设门槛,不需要 token。
import os, time
from huggingface_hub import snapshot_download
t=time.time()
p=snapshot_download("inclusionAI/UI-Venus-2-9b", local_dir="/data01/jaxan/models/UI-Venus-2-9b", max_workers=8)
print("DONE", p, f"{time.time()-t:.0f}s", flush=True)
