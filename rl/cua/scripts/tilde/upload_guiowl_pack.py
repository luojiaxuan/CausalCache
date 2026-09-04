# note (luojiaxuan): 把 GUI-Owl 成功轨迹的 tar 分片上传到 HF 私有仓,供 Tilde 侧同口径重标拉取。
# 分片而非裸文件树:18 万级 PNG 会撞 repo 提交(256/小时)与 API 限流(2026-08-21 教训)。
# 上传后用 list_repo_files 对账文件数,对不上不删本地。
import os, sys
from huggingface_hub import HfApi
os.environ["HF_HUB_DISABLE_XET"] = "1"
tok = open("/data01/jaxan/token_gavinlaw").read().strip() if os.path.exists("/data01/jaxan/token_gavinlaw") else os.environ["HF_TOKEN"]
api = HfApi(token=tok)
who = api.whoami()["name"]
repo = "gavinlaw/causalcache-mw-guiowl-success-trajs"
print("as", who, "->", repo, flush=True)
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path="/data01/jaxan/pack_guiowl", repo_id=repo, repo_type="dataset",
                  allow_patterns=["*.tar", "manifest.txt"])
files = api.list_repo_files(repo, repo_type="dataset")
local = sorted(f for f in os.listdir("/data01/jaxan/pack_guiowl") if f.endswith(".tar") or f == "manifest.txt")
remote = sorted(f for f in files if f.endswith(".tar") or f == "manifest.txt")
print("local", len(local), "remote", len(remote), "MATCH" if local == remote else "MISMATCH", flush=True)
info = api.repo_info(repo, repo_type="dataset", files_metadata=False)
print("revision", info.sha, flush=True)
