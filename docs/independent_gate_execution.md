# Independent gate 执行与交接

## 文档范围

本文只定义 selection-biased diagnostic 之后的 independent reference gate / oracle pilot 如何落地、如何
留存 artifact，以及失败后如何停止或恢复。reference 的样本、threshold、policy revision 与 outcome 已由
committed config 和 immutable manifest 冻结；oracle distance/统计仍须第二份 committed config。本文中的
尖括号字段是运行身份占位符，必须在 formal run 前替换，不授权操作者临场选择科学参数。

Independent gate 的合法输出只决定是否进入更大规模实验。它不能单独支持 terminal-success、matched-NLL
或 distilled-gate 的论文 claim，也不能把旧 diagnostic states 变成 training labels。

## Hyper00 old-pilot hardware anchor

在查看 independent split policy output 前，Hyper00 用 UI-TARS 重跑旧 9-decision artifact，结果是
9/9 parsed、4/9 executable-match，逐 decision boolean vector 与 Aries A6000 完全一致。这个 behavioral
anchor 不是 independent gate 的结果或数据来源：

| 字段 | 已验证值 |
| --- | --- |
| Git run commit | `89f4aa618ff1e74c57bef44dd69903e1c5134f7f` |
| Host | SSH `hyper00` / hostname `node-radixark-16-0000` |
| GPU mapping | physical GPU 0 / container `cuda:0` / NVIDIA H200 |
| Container image digest | `sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa` |
| Runtime | Python 3.12.3、PyTorch 2.11.0+cu130、Transformers 5.6.0、BF16 |
| Anchor policy | `ByteDance-Seed/UI-TARS-1.5-7B@683d002dd99d8f95104d31e70391a39348857f4e` |
| Old dataset | `gavinlaw/causalcache-guiodyssey-pilot-mobile@1de9c34ff029d4c01665cdaca74436ae24bff276` |
| Observed peak / generation latency sum | 17.83 GB / 30.77 s |

该锚点证明 Hyper00 H200、persistent mount、pinned UI-TARS snapshot、原 prompt/parser 与 frozen
executable equivalence 在跨芯片时保持同一 gate decision。Formal runner 强制读取
`data/results/ui_tars_hyper00_hardware_anchor/summary.json`。它不允许 independent run 复用旧 Qwen
238 tokens/image、476 tokens/event 或旧 positive states；UI-TARS 的 restoration visual cost 必须在后续
oracle config 中单独审计。

anchor 的两个 active 10 秒窗口平均利用率为 21%/18%，原因是逐 decision variable-history image
preprocessing 与短 generation。reference gate 是固定分母的短 supervised run，仍须记录 monitor；进入
coalition-scale oracle 前必须实现 batching/concurrency 与 GPU-side KL，不能把 anchor 的低利用率当作扩展
任务豁免。

## Formal run 前必须已经存在的 source of truth

开始任何 policy inference 前，Git `main` 必须已经 push 以下对象：

1. `code/configs/independent_reference_gate_v1.json`：source pool、policy/model revision、prompt/parser、
   processor、dtype、decoding、reference gate 与 oracle boundary；
2. artifact 内的 v0.4 `manifest.json`：pinned upstream/HF revision、固定 salt、trajectory-level split、
   validated action type、history mapping、文件 SHA256 和去重规则；
3. runner/tests：clean checkout、model snapshot、dataset shard、visual cost、output emptiness 和 outcome reducer
   必须 fail closed；
4. HF destination：frozen private dataset repo
   `gavinlaw/causalcache-guiodyssey-independent-mobile`，并在 config 中记录 immutable revision；
5. exact full Git SHA 与 container image digest。mutable `main`、Docker tag 或本地路径不能充当 revision。

`reference_gate` 与 `oracle_pilot` 必须在 trajectory 层不重叠，也必须排除旧 source id
`0054832199799795`。Reference gate 通过前不得运行 oracle restoration forward；reference 失败后不得换
subset、追加 rows、修改 prompt/parser/equivalence 或放宽 threshold。

## Hyper00 formal preflight

以下命令从 Mac 执行。它们只读取 host 状态；旧 snapshot 不能代替当次检查。

```bash
ssh -T -o RemoteCommand=none -o RequestTTY=no hyper00 '
hostname
uname -m
nvidia-smi
df -hT / /data01 /data02
docker info --format "{{.DockerRootDir}}"
docker ps -a --format "{{.Names}}" | sort
'
```

随后执行项目的 10 秒 continuous-zero cleanup/preflight。`--all-containers` 只会按项目 standing policy
停止完整采样窗口内始终为 0% 的 GPU container；不得借此删除文件、volume 或 image。

```bash
bash ~/.codex/skills/gpu-idle-docker-cleanup/scripts/cleanup_idle_gpu_containers.sh \
  --host hyper00 \
  --preflight \
  --all-containers
```

将 preflight 返回的至多两个 GPU ids 保存到 run note；本工作默认只使用一个。创建 container 前再次用
`nvidia-smi -i <PHYSICAL_GPU_ID>` 检查竞争，并将该物理 id 显式写入 `--gpus`。不得用
`CUDA_VISIBLE_DEVICES` 代替 Docker device binding。

```bash
ssh -T -o RemoteCommand=none -o RequestTTY=no hyper00 \
  'nvidia-smi -i <PHYSICAL_GPU_ID>'
```

## Container 与 clean checkout

`<MMDDHHMM>`、`<PHYSICAL_GPU_ID>`、`<GIT_FULL_SHA>` 和 `<CONTAINER_IMAGE_DIGEST>` 必须用当次冻结值
直接替换。Container 名只含 ownership prefix 与创建时间，不写实验名。

```bash
ssh hyper00

test "$(docker image inspect hongccc/sglang-omni:dev --format '{{.Id}}')" = \
  '<CONTAINER_IMAGE_DIGEST>'

docker run -itd \
  --shm-size 32g \
  --gpus '"device=<PHYSICAL_GPU_ID>"' \
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
  --cap-add=SYS_PTRACE \
  --name sglang-omni-jaxan-<MMDDHHMM> \
  <CONTAINER_IMAGE_DIGEST> \
  /bin/zsh
```

这里的 digest 是本地 `docker image inspect ... .Id` 返回的完整 `sha256:...` image id；若 frozen config
改为记录 registry RepoDigest，则 config 与命令必须显式区分两种字段，不能拿其中一个校验另一个。

Container 内必须只看到一张 GPU，且 `cuda:0` UUID 与 host 选中的 physical GPU UUID 相同：

```bash
docker exec sglang-omni-jaxan-<MMDDHHMM> bash -lc '
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
python3 -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
'
```

比较 UUID 时，以 host 的 `nvidia-smi -i <PHYSICAL_GPU_ID> --query-gpu=uuid` 和 container 内上条
`nvidia-smi` 为准；PyTorch 只负责验证单卡可见性与 `cuda:0` 可用。

Checkout 使用已经 push 的 detached commit。Formal output、venv、logs 与 staging 全部位于 `/data`，不写
container layer：

```bash
docker exec sglang-omni-jaxan-<MMDDHHMM> bash -lc '
test -d /data/CausalCache/.git || git clone https://github.com/luojiaxuan/CausalCache.git /data/CausalCache
git -C /data/CausalCache fetch origin main
git -C /data/CausalCache checkout --detach <GIT_FULL_SHA>
test "$(git -C /data/CausalCache rev-parse HEAD)" = "<GIT_FULL_SHA>"
test -z "$(git -C /data/CausalCache status --porcelain --untracked-files=all)"
python3 -m venv /data/.venv/causalcache-independent-gate
cd /data/CausalCache
/data/.venv/causalcache-independent-gate/bin/python -m pip install -e '.[pilot]'
cd /data/CausalCache
make test validate-contract
'
```

若 repo 需要 private Git credentials，应使用 host 已配置的 credential helper 或一次性只读 deploy
credential；credential 不得写入 Git、run argv、result JSON 或 `/data` artifact。

## Monitor

Reference gate 与 oracle 各自启动一个 monitor，log 写 Mac `/tmp`，并在 compact result 中记录其 SHA256。

```bash
bash ~/.codex/skills/gpu-utilization-monitor/scripts/monitor_gpu_utilization.sh \
  --host hyper00 \
  --container-prefix sglang-omni-jaxan \
  --min-util 90 \
  --window-seconds 10 \
  --poll-seconds 60 \
  --log-file /tmp/causalcache-independent-gate-<RUN_ID>-gpu.log
```

Monitor 报告低于 90% 时，立即检查 image decode、CPU、disk I/O 与 batching，并保持短 reference run
全程监督、记录窗口；不得把低利用率任务无人值守。coalition-scale oracle 在解决 GPU-side KL/batching 前
不得启动。只有 batch/concurrency、host path 和 physical GPU id 属于 hardware adapter；dtype、resize、
budget、distance 或 policy 参数变化均要求新 config 与新 run id。

## Formal 命令模板

下面是必须由 runner 实际 CLI 替换并由 config tests 覆盖的完整参数面。尖括号不得原样出现在 formal
argv。每一阶段使用新的空 output directory；不得让 reference 和 oracle 共写一个目录。

### 1. Reference gate

```bash
docker exec sglang-omni-jaxan-<MMDDHHMM> bash -lc '
cd /data/CausalCache
/data/.venv/causalcache-independent-gate/bin/python -m scripts.run_independent_ui_tars_reference_gate \
  --config code/configs/independent_reference_gate_v1.json \
  --dataset-tar /data/artifacts/causalcache-guiodyssey-independent-v1/data/guiodyssey-independent-00000.tar \
  --hardware-anchor-summary data/results/ui_tars_hyper00_hardware_anchor/summary.json \
  --model-dir /data/artifacts/models/UI-TARS-1.5-7B \
  --device cuda:0 \
  --output-dir /data/experiments/<RUN_ID>/reference_gate \
  --run-git-commit <GIT_FULL_SHA> \
  --container-image-digest <CONTAINER_IMAGE_DIGEST>
'
```

Runner 必须输出 `REFERENCE_GATE_PASSED`、`NO_GO_CURRENT_REFERENCE_STACK` 或 `INVALID`。只有第一种状态
允许继续；不得由人工查看 individual outputs 后决定是否继续。

### 2. Independent oracle

当前仓库尚未实现 independent oracle runner。即使 reference 通过，也必须先提交并 push 第二份 frozen
oracle analysis config：确定 exactly-20 primary matched-state selector、UI-TARS effective visual cost、distance、
trajectory-cluster bootstrap 与 GPU-side KL/batching；之后才允许读取 `oracle_pilot` policy/restoration output。
这不是一个可由操作者补命令占位符绕过的步骤。

## Artifact 打包、HF 上传与 revision 回写

先在 Hyper00 container 内从 frozen source bytes 构建 deterministic artifact：

```bash
cd /data/CausalCache
/data/.venv/causalcache-independent-gate/bin/python -m scripts.build_guiodyssey_independent \
  --config code/configs/independent_reference_gate_v1.json \
  --source-file-manifest data/manifests/independent_reference_gate_v1_source_files.json \
  --source-root /data/source/guiodyssey-independent-v1 \
  --output-dir /data/artifacts/causalcache-guiodyssey-independent-v1
```

输出目录必须为空；builder 会在 decode rows 前重算全部 source size/SHA，再生成 `manifest.json`、README 和
单个 `data/guiodyssey-independent-00000.tar`。上传前应重跑一次到另一个空目录并比较 manifest/tar SHA，
确认 byte-identical。

Reusable candidate manifest、dataset shards、raw reference/oracle records 与完整 machine-readable results
进入同一个 stable private HF dataset repo；Git 只保存 compact summary、coalition distances、失败分类和
artifact index。不得上传 full-vocabulary log-prob tensor；runner 应在 GPU 上完成 KL 归约，只保存必要的
per-state/per-coalition scalar 和 provenance。

建议 HF layout：

```text
gavinlaw/causalcache-guiodyssey-independent-mobile
  README.md
  manifest.json
  data/guiodyssey-independent-00000.tar
  runs/<RUN_ID>/reference-gate.jsonl.gz
  runs/<RUN_ID>/oracle-results.jsonl.gz
  runs/<RUN_ID>/artifact-manifest.json
```

打包器必须生成 deterministic shard，并在 `artifact-manifest.json` 中记录 source/model/Git identity、schema、
生成 argv、record count、每个文件 SHA256 和 upload status。相同输入打包两次必须 byte-identical。HF payload
manifest 不写未来 commit OID，避免 revision 自引用。

Mac 的 `~/hf_key.txt` 仅通过 stdin 使用。下面命令在 remote container 直接上传 persistent staging folder，
不会把 token 放进 argv、环境变量或远端文件：

```bash
python3 -c 'import sys; from huggingface_hub import HfApi; api=HfApi(token=sys.stdin.readline().strip()); print(api.create_repo(repo_id="gavinlaw/causalcache-guiodyssey-independent-mobile", repo_type="dataset", private=True, exist_ok=True).repo_id)' \
  < ~/hf_key.txt

ssh -T -o RemoteCommand=none -o RequestTTY=no hyper00 \
  "docker exec -i sglang-omni-jaxan-<MMDDHHMM> python3 -c 'import sys; from huggingface_hub import HfApi; api=HfApi(token=sys.stdin.readline().strip()); info=api.upload_folder(repo_id=\"gavinlaw/causalcache-guiodyssey-independent-mobile\", repo_type=\"dataset\", folder_path=\"/data/experiments/<RUN_ID>/hf_payload\", path_in_repo=\"\"); print(info.oid)'" \
  < ~/hf_key.txt
```

把 stdout 的 40-character OID 保存为 `<HF_COMMIT_OID>`。随后从 Mac 对 immutable OID 重新列文件，并校验
下载后的 manifest/shard SHA256；只验证 mutable `main` 不算完成。

```bash
python3 -c 'import sys; from huggingface_hub import HfApi; api=HfApi(token=sys.stdin.readline().strip()); print("\n".join(api.list_repo_files(repo_id="gavinlaw/causalcache-guiodyssey-independent-mobile", repo_type="dataset", revision="<HF_COMMIT_OID>")))' \
  < ~/hf_key.txt
```

校验通过后才创建不可移动的语义 tag；若 tag 已存在则失败，不覆盖：

```bash
python3 -c 'import sys; from huggingface_hub import HfApi; api=HfApi(token=sys.stdin.readline().strip()); api.create_tag(repo_id="gavinlaw/causalcache-guiodyssey-independent-mobile", repo_type="dataset", tag="<HF_TAG>", revision="<HF_COMMIT_OID>"); print("<HF_COMMIT_OID>")' \
  < ~/hf_key.txt
```

HF 完成后，Git writeback 必须包含：repo、tag、immutable OID、artifact manifest SHA256、shard SHA256、
record counts、完整生成命令、formal outcome 与 upload verification。Compact result 放
`data/results/<RUN_ID>/`，同时更新顶层 README artifact index、`docs/progress.md` 和相关执行/实验文档；然后
运行 tests，commit 并 push canonical `main`。HF OID 尚未回写或 Git commit 尚未 push 时，里程碑状态只能
写 `local_staging`，不能写 completed。

## Failure handling

Formal runner 必须从空 output directory 开始，并以临时文件加原子 rename 写 `summary.json` 或
`failure.json`。不得覆盖旧目录、删除 invalid attempt 或把两个 attempt 合并。

失败分为四类：

1. `IMPLEMENTATION_INVALID`：prefix alignment、non-finite KL、visual-cost mismatch、model/data/config/Git
   identity、output contamination、OOM、runner exception 或 provenance 检查失败。这类结果没有科学含义；
   保存 failure artifact，修复必须先 commit/push，再从新 run id 重跑同一科学 config。
2. `INFRASTRUCTURE_INTERRUPTED`：SSH/host/container/power/disk 中断，且没有完整原子 summary。保留 `/data`
   checkpoint 与 monitor log；只有 runner 明确支持且完整 run contract 相同才允许 resume，否则使用新目录。
3. `NO_GO_CURRENT_REFERENCE_STACK`：reference gate 在完整冻结分母上合法失败。立即停止 oracle，不换
   subset/policy/prompt/parser/threshold；照常打包上传并回写 Git，作为有效 negative result。
4. 科学 `GO_TO_FULL_EXPERIMENT`、`NO_GO_ORACLE` 或 `INCONCLUSIVE`：只有 oracle 全部 states、预算、seed
   与 baseline 完成且 reducer 验证通过后才能产生。任何缺 state、缺 seed 或提前停止均不得归入此类。

OOM 或吞吐问题不能通过临场改 dtype、resize、history、budget 或 coalition 数来“恢复”。允许的 batching/
concurrency 修复也必须保持逐 coalition 数值语义，并在新 attempt metadata 中记录。若 monitor 持续低于
90%，先停止 formal run；不得因为旧 pilot 曾有 0% window 就豁免。

每个失败 attempt 至少保存：run id、UTC start/failure time、完整 argv、Git/config/model/data/image identity、
host/GPU/driver/runtime、已完成 record ids、exception/failure classification、monitor log SHA256、local staging
path、预期 HF destination 与 upload status。敏感 token 和 full logits 不得进入 failure artifact。

## 合作者交接清单

交接时必须一次性给出以下信息；只发共享机器路径或聊天截图不算完成：

- clean Git full SHA 和 canonical remote branch；
- independent config path/SHA256、candidate manifest HF repo/revision/SHA256；
- model repo/revision、local snapshot path 与 `.snapshot.json` 校验状态；
- container image digest、host、physical GPU id、container `cuda:0` UUID；
- reference/oracle run ids、完整 argv、output directories、monitor log 与当前 classification；
- HF repo/tag/immutable OID、artifact manifest/shard SHA256 与 upload verification；
- Git compact result、progress/writeback commit 和 pushed status；
- 当前 blocker 与唯一下一动作，例如“reference gate 未运行”“reference 合法失败，禁止 oracle”或
  “oracle artifact 已上传，等待 Git writeback”。

最终 Definition of Done 仍是：formal result verified、reusable artifacts 位于 HF immutable revision、轻量
结论和链接写入 Git、tests 通过、commit 已 push `main`。任一项缺失都必须显式交接为 unfinished。
