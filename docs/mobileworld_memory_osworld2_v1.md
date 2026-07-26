# MobileWorld-Memory 与 OSWorld 2.0 接入 v1

## 目标与边界

本线从重新同步的 `main@23c0fb4` 创建分支
`luojiaxuan/mobileworld-memory-osworld2`，新增两个 benchmark substrate：

1. MobileWorld 使用冻结 GUI-Owl、显式 recent-at-most-B4 memory、单 GPU 共享 policy
   server 与多 Android emulator 动态任务队列；
2. OSWorld 2.0 直接采用官方 task-construction phenomenon 标注定义 memory split，不从
   policy 失败轨迹事后解释 memory dependency。

本步不训练或修改 GUI-Owl，不接触 sealed AndroidWorld test split，也不根据 benchmark
结果修改任务分母、memory arm、budget、prompt、parser 或 action schema。

## 固定上游与分母

### MobileWorld

- repository：`https://github.com/Tongyi-MAI/MobileWorld`
- revision：`8ae506487bf87785292d6cad101c49955d704d39`
- environment image：
  `ghcr.io/tongyi-mai/mobile_world@sha256:b680380eac98a7ad064707f9653772af18554d201a3e6e7cf8f15d58cdc73240`
- task/runtime source：上述 Git revision 的 `src/` 只读挂载到 image 的
  `/app/service/src`；fleet manifest 同时锁定 image digest 与 source revision
- manifest：[`data/manifests/mobileworld_memory_split_v1.json`](../data/manifests/mobileworld_memory_split_v1.json)

AST inventory 固定 201 个 task class、20 个 app。按官方 interface tag 分为 GUI-only 117、
MCP 38、user-interaction 44、同时需要二者 2。冻结 GUI-Owl 只有 mobile GUI action
tool，因此正式 runnable denominator 是 117，而不是把缺失 MCP/user tool 的 84 个任务算成
policy failure。

在 117 个 GUI-only tasks 内，按 task class 预先声明的 `app_names` 构造：

| split | 数量 | 构造规则 |
|---|---:|---|
| `cross_app_memory_candidate` | 62 | `app_count >= 2` |
| `single_app_control` | 55 | `app_count == 1` |

这是一项 pre-execution、可复现的结构候选定义；cross-app 本身不等同于已经证明的
memory dependency。未来若需要更强 MobileWorld-Memory split，应在 task generator 中加入
可控的 hidden-state write/read、延迟长度与反事实恢复字段，不能按本次失败案例追认标签。

### OSWorld 2.0

- repository：`https://github.com/xlang-ai/OSWorld-V2`
- release：`osworld-v2-2026.06.24`
- tag / code revision：
  `v2026.06.24@2b9b7b4eb73243d557bdbf2998fe18d8e18e19c6`
- task count：108
- manifest：[`data/manifests/osworld_v2_memory_split_v1.json`](../data/manifests/osworld_v2_memory_split_v1.json)

split 完全来自官方 release 的 task-construction phenomenon：

| split | 数量 | 构造规则 |
|---|---:|---|
| `memory_core` | 43 | `implicit_state_inference` |
| `memory_stress_union` | 65 | dynamic / cross-source / implicit 三者并集 |
| `non_memory_control` | 43 | 上述并集的补集 |

其中 dynamic=10、cross-source=46、implicit=43、三者交集=8。2026-07-25 已确认当前
Hugging Face 用户可读取 `xlangai/osworld_v2_tasks@v2026.06.24` 与
`xlangai/osworld_v2_assets_gated@v2026.06.24`；对应 immutable revisions 分别为
`e7996f4cc850be108e510bd8433c63ee7b8303dd` 与
`bee070ab3d61c74786622c60da45ae9a6e47c54e`。正式执行仍严格 fail closed：
task classes 必须完整 108 个且本地 asset root 必须存在。
不得以旧 OSWorld task、空 asset 或手写替代 task class 冒充 OSWorld 2.0 结果。
asset root 由 committed config 的 `execution.assets_root` 或显式 `--assets-root` 传入；
runner 只在 worker 内把该显式值翻译为上游所需的 `OSWORLD_FILE_BASE_URL`，不读取 ambient
environment override 来改变 gate。

OSWorld 2.0 Docker provider 复用本项目已验证的 runtime adapter：
`DNSMASQ_OPTS=--no-resolv --no-poll --server=127.0.0.11`、sparse container port
snapshot 与 180 秒全局 port-allocation lock timeout。科学任务、policy 与 evaluator 不变；
该 adapter 只消除多 environment 并发启动时的 DNS watcher、list→inspect race 和 10 秒锁超时。

## 单 GPU、多模拟器拓扑

```text
Aries GPU 1
└── canonical compute container: sglang-omni-jaxan
    ├── one frozen GUI-Owl policy process
    │   └── ThreadingHTTPServer + single inference lock
    ├── MobileWorld runner dynamic queue
    └── inner Docker daemon on persistent /data
        ├── emulator exception container 0
        ├── emulator exception container 1
        └── ... emulator exception container N-1
```

Android emulators只使用 CPU/KVM；GPU 只加载一份冻结 policy。多个 environment 并发 reset、
database/local-storage/callback verification 与 step wait，在一个串行 GPU policy 后面隐藏环境
等待。capacity 依次测试 N=1/2/4/8，每点从 117 task roster 等距固定 16 个 task、每 task
最多一步、infra failure 最多自动重试 2 次；以完整 16/16 denominator 的 throughput
拐点选择完整 117-task run 的 N。

Aries 宿主 Docker root 只剩约 8.4 GiB，不能直接拉取 MobileWorld 大镜像。canonical
container 已确认 privileged、可见 `/dev/kvm`，并把 `/mnt/data6/jiaxuanluo` 持久挂载为
`/data`（约 494 GiB available）。因此 emulator image/layers 放入 `/data` 上的 inner
Docker data-root；不得 prune 宿主 Docker，也不得停止或删除既有他人/历史容器。

实机 preflight 发现官方 GHCR digest 内的 task source 早于当前 Git revision：image 内
`ThanksgivingPrepTask` SHA256=`ba472079...d694`，缺少 `reset_chrome` import，而 frozen
Git source SHA256=`ee800d1f...f74c`。因此 image 只提供 Android/backend/system layers，
task/runtime Python source 必须来自 pinned Git 的 read-only `src/` mount；不允许让 image
内旧 source 静默决定 denominator。

## 可复现入口

生成与验证两个 construction manifests：

```bash
PYTHONPATH=code python3 code/scripts/build_mobileworld_memory_plan.py \
  --mobileworld-root /path/to/MobileWorld \
  --expected-revision 8ae506487bf87785292d6cad101c49955d704d39 \
  --output data/manifests/mobileworld_memory_split_v1.json

PYTHONPATH=code python3 code/scripts/build_osworld_v2_memory_plan.py \
  --osworld-root /path/to/OSWorld-V2 \
  --output data/manifests/osworld_v2_memory_split_v1.json
```

启动一组带 identity audit 的 MobileWorld emulator exception containers：

```bash
DOCKER_HOST=unix:///data/run/mobileworld-docker.sock \
PYTHONPATH=code python3 code/scripts/launch_mobileworld_environments.py launch \
  --count 8 \
  --image ghcr.io/tongyi-mai/mobile_world@sha256:b680380eac98a7ad064707f9653772af18554d201a3e6e7cf8f15d58cdc73240 \
  --name-prefix sglang-omni-jaxan-mw1- \
  --port-seed '<run-secret>' \
  --source-root /data/upstreams/MobileWorld \
  --expected-source-revision 8ae506487bf87785292d6cad101c49955d704d39 \
  --output /data/runs/mw-memory-v1/environment-fleet.json
```

共享 policy 与 full run：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=code python3 \
  code/scripts/serve_mobileworld_gui_owl_policy.py \
  --model-dir /root/.cache/huggingface/hub/models--mPLUG--GUI-Owl-1.5-8B-Instruct/snapshots/06d5faecff74840bab2be2425e9c42667a5d04fc \
  --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
  --device cuda:0 --port '<irregular-high-port>' --visual-tokens 2560

PYTHONPATH=code uv run --project /data/upstreams/MobileWorld \
  python code/scripts/run_mobileworld_gui_owl.py \
  --repository-root /data/repo/CausalCache \
  --mobileworld-root /data/upstreams/MobileWorld \
  --fleet-manifest /data/runs/mw-memory-v1/environment-fleet.json \
  --policy-endpoint http://127.0.0.1:'<irregular-high-port>' \
  --output-root /data/runs/mw-memory-v1/full \
  --profile full --num-envs '<capacity-knee>'
```

每次 execution record 保存 Git/upstream revision、完整 argv、config/manifest SHA256、image
identity、task roster SHA256、GPU 前后快照、policy counters、结果数量和 wall time。raw
screenshots/trajectories 只保留在个人 persistent storage，不进入 Git；Git 只回写轻量
summary。正式清理只读取本次 fleet manifest，逐个核验 container ID 后停止该 run 创建的
exception containers，保留 canonical compute container。

所有 memory arms 都保留完整的低带宽 action/`screen_changed` event summaries；B0/B1/B2/B4
预算只控制恢复的历史截图数量。这与 CausalCache 现有 low-fidelity-summary + selective visual
restoration contract 一致，不能把 B0 解释为完全无文本历史。

## 当前状态

- MobileWorld 与 OSWorld 2.0 construction manifests：已生成并通过 CPU contract tests；
- shared frozen GUI-Owl policy、MobileWorld upstream agent、multi-environment runner：
  已实现；
- OSWorld 2.0 runner：已实现；gated tasks/assets 权限已批准，固定 release 下载与
  5-GPU full run 准备中；
- MobileWorld capacity：1/2/4/8 environments 的 16-task wall time 分别为
  `383.715/192.133/102.320/62.540 s`，8 env 达到 `921.010 tasks/hour`；
- MobileWorld full campaign：先用 Aries GPU 1 + 8 env 得到 52 个结果，再在 GPU 0
  释放后把剩余 65 task 冻结为 33/32 两个互斥 shard、各用 8 env。总 makespan
  `22,863.108 s = 6:21:03.108`；最终 114/117 有结果、27 success，3 个 policy-invalid
  task 在三次尝试后仍 missing。observed mean=`0.236842`，missing 计 0 的严格
  117-task mean=`0.230769`。详细结果见
  [`data/results/mobileworld_frozen_gui_owl_benchmark_v1/`](../data/results/mobileworld_frozen_gui_owl_benchmark_v1/README.md)。

## Source of Truth

- code/config/docs/manifests：本 Git 分支；
- reusable upstream datasets/assets：各官方 repository/release，不复制到本项目 HF；
- run root：Aries `/mnt/data6/jiaxuanluo/runs/mw-memory-v1`（container 内 `/data/runs/mw-memory-v1`）；
- raw trajectories：上述 persistent run root，状态 `LOCAL_PRIVATE_RAW_TRACE`；
- lightweight benchmark summary：
  [`data/results/mobileworld_frozen_gui_owl_benchmark_v1/`](../data/results/mobileworld_frozen_gui_owl_benchmark_v1/README.md)。
