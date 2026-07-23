# OSWorld runner v1

## 当前状态

`luojiaxuan/osworld-runner` 已接通 OSWorld 官方 `DesktopEnv` 的四个核心接口：

```text
task JSON -> reset(task_config) -> screenshot observation
          -> typed desktop action -> step(pyautogui)
          -> evaluate() -> task score
          -> close()
```

runner 固定 OSWorld revision `b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf`，实际读取官方
`evaluation_examples/test_all.json` 的 10 个 domain / 369 个 tasks。revision、task id、action schema 或
response schema 漂移都会 fail closed。

Docker provider 额外固定 `DNSMASQ_OPTS=--no-resolv --no-poll --server=127.0.0.11`。原因是共享 Linux
host 上 dnsmasq 的 inotify instance 可能耗尽，官方默认会退化到不支持 host port forwarding 的 user-mode
network；adapter 禁止 resolv-file watcher，并显式把 Docker embedded DNS 保留为 upstream。该修复只作用于
OSWorld VM container，不修改 host sysctl。

这是可运行的 benchmark substrate，不是 CausalCache 的跨平台科学结果。目前已实现 `summary`、`recent`、
`full` 三个 memory arms 和 policy HTTP boundary；learned selector、GUI-Owl desktop checkpoint 以及正式
OSWorld task roster 尚未接入。

Hyper01 和 H100 live smoke 均完成真实 `reset -> screenshot -> WAIT -> DONE -> evaluate -> close`，随后用相同
命令验证 `resumed_skips=1`。轻量证据见
[`../data/results/osworld_runner_v1_smoke/`](../data/results/osworld_runner_v1_smoke/)和
[`../data/results/osworld_runner_v1_h100_smoke/`](../data/results/osworld_runner_v1_h100_smoke/)。score=`0.0` 是
scripted executor smoke 的预期值，不是 policy performance。

H100 的 Intel 8462Y+ / Linux 5.15 host 在官方 Docker image 默认 `CPU_MODEL=host` 时会停在 GRUB/early boot，
5 分钟内不能提供 screenshot endpoint；同一 qcow2、Docker digest 与 KVM 改为 `CPU_MODEL=qemu64` 后约 12 秒
ready。runner 因此提供显式 `--docker-cpu-model`，该值进入 result provenance。Hyper01 不需要 override；不得
把 H100 的 host-specific execution adapter 当作模型或 benchmark 参数。

## 为什么不直接复用 AndroidWorld runner

AndroidWorld action 是 mobile JSON action；OSWorld executor 接受 desktop PyAutoGUI。为避免让模型输出任意
Python，OSWorld adapter 先验证一个受限 action mapping，再由 runner 渲染成 PyAutoGUI：

- `move`、`click`、`double_click`、`right_click`、`drag`；
- `scroll`、`type_text`、`press`、`hotkey`；
- `wait`、`done`、`fail`。

坐标、按钮、点击次数、duration、文本长度和字段集合均在执行前检查。额外 `code` 字段、越界坐标或未知
action 会被拒绝，模型响应不会作为原始 Python 直接透传。

## 安装与 preflight

OSWorld 依赖较重且固定 Python 3.12；建议为官方 checkout 单独建环境，再以 `--no-deps` 安装本项目，避免
两边的 Transformers/PyTorch 约束互相覆盖：

```bash
git clone https://github.com/xlang-ai/OSWorld.git /data/jaxan/OSWorld
git -C /data/jaxan/OSWorld checkout b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf
python3.12 -m venv /data/jaxan/venvs/osworld
/data/jaxan/venvs/osworld/bin/pip install -r /data/jaxan/OSWorld/requirements.txt
/data/jaxan/venvs/osworld/bin/pip install --no-deps -e /data/jaxan/OSWorld
/data/jaxan/venvs/osworld/bin/pip install --no-deps -e /data/jaxan/CausalCache

/data/jaxan/venvs/osworld/bin/python -m scripts.run_osworld \
  --repository-root /data/jaxan/CausalCache \
  --osworld-root /data/jaxan/OSWorld \
  --preflight
```

成功输出必须含 `VALID_OSWORLD_PREFLIGHT`、`task_count=369` 和 exact OSWorld revision。本机只想验证 Git
revision、inventory 与 task JSON，不启动 emulator 时可运行：

```bash
PYTHONPATH=code python3 -m scripts.run_osworld \
  --repository-root . \
  --osworld-root /absolute/path/to/OSWorld \
  --dry-run \
  --domain chrome \
  --task-id <TASK_ID>
```

## 最小真实环境 smoke

官方 Docker provider 需要可用 Docker daemon；Linux 有 `/dev/kvm` 时性能更好。下面的 smoke 会执行
`WAIT -> DONE`，验证 reset、截图、action executor、evaluator 和结果落盘，不验证 policy 能力：

```bash
/data/jaxan/venvs/osworld/bin/python -m scripts.run_osworld \
  --repository-root /data/jaxan/CausalCache \
  --osworld-root /data/jaxan/OSWorld \
  --domain chrome \
  --task-id <TASK_ID> \
  --provider docker \
  --docker-cpu-model qemu64 \
  --path-to-vm /data/jaxan/osworld/Ubuntu.qcow2 \
  --output-root /data/jaxan/osworld/smoke \
  --cache-dir /data/jaxan/osworld/cache \
  --policy scripted \
  --scripted-actions data/fixtures/osworld_scripted_smoke_actions.json \
  --max-steps 2
```

`--docker-cpu-model qemu64` 仅在已观察到默认 host CPU early-boot failure 的 H100 profile 使用；其他 host 默认
保持官方 `host` model。OSWorld Docker VM 是 KVM/CPU workload，这个 smoke 没有分配 H100 GPU。

## Policy HTTP contract

正式 agent 以 `--policy http --policy-url http://HOST:PORT/act` 接入。每一步 POST 包含：

- instruction、domain、task id 和当前 screenshot；
- 全部历史 event 的结构化摘要；
- memory arm 选中的 post-action screenshots；
- screen size 与受限 action 类型。

服务端返回：

```json
{
  "schema_version": "causalcache.osworld.policy_response.v1",
  "action": {
    "type": "click",
    "x": 960,
    "y": 540,
    "button": "left"
  }
}
```

这条边界允许后续分别接 GUI-Owl desktop runtime、recent baseline 和 learned selector，不把 environment
runner 锁死到单一模型服务。

## 断点与并行

每个 task 的唯一完成标记是：

```text
<output-root>/<domain>/<task-id>/result.json
```

已完成 task 在重启后直接跳过。进行中的 episode 每一步原子更新 `attempts/<attempt-id>/checkpoint.json` 并
保存 screenshot；进程中断时保留 `failure.json` 或已有 checkpoint。由于 OSWorld VM state 不能可靠跨进程
恢复，未完成 task 下次从 clean `reset(task_config)` 重跑，不伪造 mid-episode resume。

批量运行用 `--all`，并以 `--num-shards N --shard-index I` 做确定性 task-level 分片。多个 runner 必须使用
不同 shard index；输出目录共享时依靠 task completion marker 续跑。

## Source of Truth

| 内容 | Git 位置 |
| --- | --- |
| environment/action/policy/episode adapter | `code/causalcache/osworld.py` |
| CLI | `code/scripts/run_osworld.py` |
| pinned config | `code/configs/causalcache_osworld_runner_v1.json` |
| focused tests | `code/tests/test_osworld.py` |
| scripted executor smoke actions | `data/fixtures/osworld_scripted_smoke_actions.json` |
| live smoke summary | `data/results/osworld_runner_v1_smoke/` |
| H100 portability smoke summary | `data/results/osworld_runner_v1_h100_smoke/` |
| concurrent benchmark launcher | `code/scripts/run_osworld_multienv.py` |
| H100 benchmark profile | `code/configs/causalcache_osworld_benchmark_h100_v1.json` |
| acceleration design and no-GDrive roster | `docs/osworld_benchmark_acceleration_v1.md` |

真实 rollout screenshots、trajectory traces 和 recordings 只写 persistent storage；形成可复用 benchmark artifact
后上传 Hugging Face，Git 仅保存 manifest、revision 和轻量 summary。
