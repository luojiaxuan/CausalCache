#!/usr/bin/env python3
"""把 desktop-did-corpus-v7pool(语料 + 帧子集)推到 HF 私有数据集仓库。

# note (luojiaxuan): 数据搬运走 HF 而不是经本机中转(全局规则)。tilde 的登录节点
# 与 hyper 主机互不可达,而两边都能到 huggingface.co —— HF 就是它们之间唯一的
# 直连通路,顺带满足"可复用产物以 HF 为准"的 source-of-truth 约定。
#
# 语料与帧放同一个 repo,因为它们**必须配套**:corpus 的 image_relpaths 直接索引
# frames 下的相对路径,分开发布会让下游少拿一半就跑不起来。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO = "gavinlaw/causalcache-desktop-did-v7pool"

CARD = """# CausalCache desktop DiD corpus v7pool(私有)

固定预算替换语料 + 每组 K=3 个非 witness 候选臂,供 `did_pool_rank` 目标使用。

| 项 | 值 |
|---|---|
| 组数 | 1,645(满池要求淘汰了 corpus-v4 的 25%) |
| 样本 | 18,095 = 1,645 × 11 臂(R0/RA/S0/SA/WA + C1..C3 各 active/bypass) |
| schema | `causalcache.desktop_did_sample.v3` |
| 帧 | 5,570 张唯一图片,`frames/frames_subset.tgz`,按 `image_relpaths` 的相对路径组织 |
| 生成命令 | `build_desktop_did_corpus_v4.py --b-values 1 2 4 8 --candidate-pool-size 3 --seed 20260726` |
| 上游 manifest | `agentnet_screening_manifest_ubuntu_v1.jsonl`(AgentNet/OpenCUA) |

**用法**:`corpus/` 作 `--dataset-root`;把 `frames/` 解包后软链到
`corpus/ubuntu_images`。K=0 时同一脚本逐字节复现 corpus-v4(已验:150 条 0 差异)。

**为什么要这份语料**:原 DiD 目标里只有 `L_gain = [m - A_c]_+` 是绝对量,
它是"给所有集合一起抬高"的唯一激励来源(实测占 v4 适配器效应的 77%)。
候选臂让目标可以只用差来表达,均匀抬升在数学上得不了分。
详见仓库 `data/results/hgkv_v6_capsweep_v1/PREMISE_REFUTED.md`。
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--corpus", type=Path,
                   default=Path("/data/desktop-did-corpus-v7pool"))
    p.add_argument("--frames-tgz", type=Path,
                   default=Path("/data/frames_subset.tgz"))
    args = p.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    print("repo 就绪:", args.repo, flush=True)

    api.upload_file(path_or_fileobj=CARD.encode("utf-8"), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset")
    api.upload_folder(
        folder_path=str(args.corpus), path_in_repo="corpus",
        repo_id=args.repo, repo_type="dataset",
        # ubuntu_images 是软链到 26 GB 全量帧库的,绝不能跟着上传
        ignore_patterns=["ubuntu_images", "ubuntu_images/**",
                         "build.log", "BUILD_DONE"],
    )
    print("corpus 上传完成", flush=True)

    if args.frames_tgz.exists():
        api.upload_file(path_or_fileobj=str(args.frames_tgz),
                        path_in_repo="frames/frames_subset.tgz",
                        repo_id=args.repo, repo_type="dataset")
        print("frames 上传完成", flush=True)

    print("REVISION", api.repo_info(args.repo, repo_type="dataset").sha, flush=True)


if __name__ == "__main__":
    main()
