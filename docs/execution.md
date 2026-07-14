# 跨芯片执行与团队交接

## 目标

合作方可以使用不同 GPU，但必须执行同一套实验语义。硬件 adapter 只允许改变吞吐相关参数；policy、
data、preprocessing、decoding、budget、seed 与 gate 均由 Git/HF revision 固定。

## 三层 source of truth

1. Git `main`：代码、config、脚本、论文、轻量 summary、progress 与本执行文档；
2. Hugging Face：全部 reusable datasets、raw data artifacts、model weights、checkpoints、adapters；
3. Mac/Hyper01/Aries 本地盘：cache、staging 和 active run，可随时重建，不是 canonical source。

Mac 上的 HF token 位于 `~/hf_key.txt`。该文件不得复制进仓库、命令日志、result JSON 或 Docker
image；`.gitignore` 也显式拒绝 `hf_key.txt` 与 `.secrets/`。优先在 Mac 已认证的 `hf` CLI 上执行
HF create/upload/tag。远端确需访问 private repo 时，由操作者通过安全通道单独登录，不把 token
作为实验参数或 shell 环境变量记录。

## 执行位置

| 工作 | 默认位置 | 备选 | 原因 |
| --- | --- | --- | --- |
| 文档、Git、LaTeX、单测、轻量数据处理 | Mac | Aries/Taurus | 不占共享 GPU |
| 大模型 inference、attribution、gate training | Hyper01 H200 | B200 | H200 吞吐和显存明显优于 A6000 |
| AndroidWorld emulator + policy rollout | Aries A6000 | Hyper01 H200（待解锁） | Aries stack 已验证；Hyper01 先解决 Docker root 容量并重做 environment smoke |
| 小模型 smoke、sample-level debug | Aries/Taurus A6000 | Hyper01 | 避免为小任务占用 H200 |

2026-07-14 实测 Hyper01：8×NVIDIA H200（每卡 143,771 MiB）、x86_64、`/dev/kvm` 可用；
`/data01` 约 1.8T 可用，`/data02` 约 975G 可用，而根分区只剩约 7G。Docker root 是
`/var/lib/docker`，当前有 `hongccc/sglang-omni:dev`，但没有约 13.4G 的
`causalcache-androidworld` image。禁止用全局 Docker prune 腾空间；在 Docker root 被管理员迁移或
明确释放足够空间前，Hyper01 只承担已有 image 可完成的 policy/offline 工作，closed-loop MVP 留在
Aries。任何 repo、virtualenv、log、checkpoint 与 cache 都不得写入根分区或容器层。

## 芯片无关的实验逻辑

以下字段发生变化时必须视为新实验 config，不能解释成“换机器”：

- Git commit、HF dataset/model repo 与 exact revision；
- task split、instance records hash、seed、step budget；
- prompt、action grammar、executable equivalence、parser；
- dtype、processor revision、image resize/grid、visual token budget、history packing；
- decoding 参数、teacher validation、attribution sampling、selector threshold；
- benchmark/server image digest 与 reward implementation。

允许作为 host adapter 改变的只有：物理 GPU id、单/双卡映射、batch/concurrency、emulator worker
数、cache/staging 绝对路径。改变这些参数后，语义输出仍需通过固定 smoke。

## 每次 GPU job 的 preflight

从 Mac 非交互检查 Hyper01：

```bash
ssh -T -o RemoteCommand=none -o RequestTTY=no hyper01 '
hostname
uname -m
test -e /dev/kvm && ls -l /dev/kvm
nvidia-smi
df -hT / /data01 /data02
docker ps -a --format "{{.Names}}" | sort
'
```

启动任何 GPU job 前必须额外执行项目约定的 10 秒 continuous-zero idle cleanup，最多选择 2 张
空闲 GPU，并将选中的 id 显式写进 Docker `--gpus` 或 Python `--device cuda:N`。长任务同时启动
GPU utilization monitor；低于 90% 时检查 batching、I/O、ADB 等待或减少 GPU 数，不能无人值守地
低效运行。

## Hyper01 容器约定

默认 image 为 `hongccc/sglang-omni:dev`。下面示例使用 GPU 0；必须用 preflight 的实际选择替换。
科学参数不通过环境变量传递；固定的 cache 环境变量只负责把基础设施写入持久盘。

```bash
ssh hyper01

docker run -itd \
  --shm-size 32g \
  --gpus '"device=0"' \
  -v /data01/cache/huggingface:/root/.cache/huggingface \
  -v /data02/jaxan:/data \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
  -e XDG_CACHE_HOME=/data/.cache \
  -e PIP_CACHE_DIR=/data/.cache/pip \
  -e WANDB_DIR=/data/wandb \
  -e TMPDIR=/data/tmp \
  --ipc=host \
  --ulimit nofile=65536:65536 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --privileged \
  --cap-add=SYS_PTRACE \
  --name sglang-omni-jaxan-$(date +%m%d%H%M) \
  hongccc/sglang-omni:dev \
  /bin/zsh
```

容器内统一使用：

```bash
git clone https://github.com/luojiaxuan/CausalCache.git /data/repo
git -C /data/repo fetch origin main
git -C /data/repo checkout --detach <GIT_COMMIT>
test -z "$(git -C /data/repo status --porcelain)"
python3 -m venv /data/.venv/causalcache
/data/.venv/causalcache/bin/python -m pip install -e /data/repo
cd /data/repo
/data/.venv/causalcache/bin/python -m scripts.validate_contract \
  --config code/configs/phase0_contract.json \
  --decision data/fixtures/validated_decision.json
```

正式 run 必须 checkout 已 push 的 exact commit 并保持 clean detached worktree；开发阶段的同一 topic
可以复用 `/data/repo` 后回到 `main` 执行 `git pull --ff-only origin main`，不要重复 clone 到多个散乱
目录。当前 validated AndroidWorld 路径是在 Aries 同一 host 上运行 emulator HTTP containers 与本地
policy runtime；尚未验证“Aries emulator + Hyper01 policy”的拆机拓扑。若以后实现 policy RPC 或
受控 tunnel，必须作为独立里程碑测试并记录，不得把它当成现有 runner 已支持。worker 数可以因机器
改变，但 validation plan、instance denominator 与失败处理不能改变。

62-instance validation 的命令必须显式传入
`--early-stop-when-success-is-mathematically-impossible`。每个 episode 作为原子 checkpoint 写盘后，runner
用固定 62 分母计算 `observed_success + unobserved_count`；只有该上界严格小于达到
50% 所需的 31 个 success 时才停止分配新 episode。已在途的 worker 必须完成 score 和
tear-down，所以最终 checkpoint 可比首次触发边界多最多 `worker_count-1` 条。非 resume 运行
必须使用空 output directory；resume 前必须校验 plan index、instance 与 filename。
每条 episode 还必须带 `run_contract`，固定 Git commit、model snapshot identity、runtime/processor、
visual preprocessing、generation、plan hash 和 server digest。resume 只能接受完整 contract 相同的
checkpoint，防止 Think 误复用 Instruct 的旧 episode。CLI 传入的 `--run-git-commit` 必须是
full SHA，且 runner 会校验它等于 clean checkout 的实际 HEAD。

## Aries/Taurus fallback adapter

Aries 是当前 AndroidWorld closed-loop MVP 的 validated host。每次运行仍需重新检查所有本地盘，不能
把上次使用的 `/mnt/data6/jiaxuanluo/causalcache` 当作永久最佳路径：

```bash
ssh -T -o RemoteCommand=none -o RequestTTY=no aries '
hostname
uname -m
test -e /dev/kvm && ls -l /dev/kvm
nvidia-smi
df -hT / /mnt/data /mnt/data2 /mnt/data3 /mnt/data4 /mnt/data5 /mnt/data6 /mnt/data7
docker ps -a --format "{{.Names}}" | sort
'
```

根据当次 `df -hT` 选择一个 local `/mnt/data*` personal directory，显式挂载为容器 `/data`；HF cache
也选择有空间的 local disk，不能写 Aries 根盘或 Taurus/Aries cross-mount 做重 I/O。Taurus 使用相同
原则，适合小模型 smoke、数据处理和 sample-level evaluation。A6000 默认单卡，只有明确吞吐理由才用
第二张卡。

同一个 attribution-label dataset 只能指定一种 canonical hardware/runtime；默认在 Hyper01 H200
生成，Aries A6000 只做固定 canary 或 closed-loop MVP，不能把两种芯片生成的 labels 混进同一 train
split。跨芯片 replication 单独记录和报告。

## HF token 的安全传递

Mac 的 `~/hf_key.txt` 只作为 stdin 输入，不能出现在 argv、环境变量、远端文件或日志。若 remote
container 必须直接上传 private artifact，可使用单次 stdin API 调用；下面命令中的 repo/path 必须
显式替换，输出只保留 immutable HF commit OID：

```bash
ssh -T -o RemoteCommand=none -o RequestTTY=no hyper01 \
  "docker exec -i <CONTAINER_NAME> python3 -c 'import sys; from huggingface_hub import HfApi; api = HfApi(token=sys.stdin.readline().strip()); info = api.upload_folder(repo_id=\"gavinlaw/<HF_REPO_ID>\", repo_type=\"dataset\", folder_path=\"/data/experiments/<RUN_ID>\", path_in_repo=\"data\"); print(info.oid)'" \
  < ~/hf_key.txt
```

上传后从 Mac 用已认证的 HF client 验证 repo visibility、文件清单和 revision，再把 exact revision
写回 Git。不得把 token 复制到 `/data` 作为长期 secret。

## 跨芯片一致性 smoke

新 host 或新镜像进入正式实验前，在同一个 committed fixture 上比较：

1. model/data revisions 与 SHA256；
2. processor `image_grid_thw`、effective visual tokens、input token count；
3. finite logits、唯一 action parse、canonical executable action；
4. deterministic decoding 下的 raw output；若浮点 logits 或 natural-language description 有微小差异，
   记录差异，不放宽 tool-call grammar 或 executable equivalence；
5. AndroidWorld reset、execute、score、tear-down 与 server image digest。

只有 smoke 通过后才启动完整 rollout。H200 与 A6000 的 latency、peak memory 可以不同；action、grid、
token budget、task reward 与 gate decision 不应因芯片不同而改变。

GUI-Owl Think 在同一 H200 上的两次 `do_sample=false` 运行已观察到一个 Action description
句点的引号内/外位置差异，tool-call JSON 与 canonical action 不变。因此 executable gate 可按
canonical action 执行；但后续 teacher-forced token distance 不得假定 raw generation byte-identical，必须
固定 action token boundary 并报告 repeat-forward variance。

## Run metadata 与回写

每个 material run 至少记录：Git commit、完整命令、host、GPU model/id、driver、CUDA、PyTorch、
Transformers、dtype、Docker image/digest、model/data HF revision、seed、worker 数、起止时间、结果与
failure classification。字段契约见 `code/configs/run_manifest.schema.json`。轻量 summary 写入
`data/results/<run>/`；raw traces/checkpoints 先上传 HF，再把 repo/tag/revision 写回 README、manifest
与 `docs/progress.md`。
没有 RNG 参数的 deterministic runner 必须记录 `seed=null`，不得为了满足 metadata 伪记一个
未实际设置的 seed。

一个里程碑的完成顺序固定为：

```text
preflight → run/smoke → verify → upload reusable artifacts to HF
→ update README/docs/data summary → test → commit → push main
```

任何一步尚未上传或 push，都必须在 README/docs 标记为 local staging，不能口头视为完成。
