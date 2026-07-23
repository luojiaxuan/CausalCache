# AndroidWorld environment/reward smoke

## 结论

**通过。** Pinned AndroidWorld emulator 能在 Aries 的 KVM 环境启动，116 个 task types 可读取，
validation seed suite 可重建；`SystemWifiTurnOn` 经同一 HTTP executor 完成设置导航后，score 从
`0.0` 变为 `1.0`，最后成功 tear down。

这一步只验证 environment、executor 与 reward 链路，不构成 GUI-Owl closed-loop success 证据。

## 固定环境

- AndroidWorld / agent code：`X-PLUG/MobileAgent@11cea575561fb7800b5fb6b6cafa56f7a91de11f`
- Canonical benchmark reference：`google-research/android_world@3e50888527ef9f29b9157ecd537e408008bb1c85`
- Host：Aries，x86_64，`/dev/kvm`
- Emulator：Pixel 6，Android API 33，Google APIs x86_64
- Docker image：`causalcache-androidworld:11cea575`
- Image digest：`sha256:24e3b08f8b757fdf98c97cfb097d07036f8ee291b5e9cbb7d1744a4c6abe3a16`
- Image size：13,382,312,586 bytes
- Runtime container：`sglang-omni-jaxan-07141346`，仅作为 Aries 可重建 cache

上游 Dockerfile 的三个 build-tooling compatibility fix 由
`code/scripts/prepare_androidworld_dockerfile.py` 严格生成：替换已移除的 Java base image、固定 uv
`0.11.28`，并为 Python 3.11 非隔离构建预装 `wheel==0.45.1` 与
`grpcio-tools==1.71.0`。未修改 task、reward、agent、prompt 或 AndroidWorld Python 源码。

## Smoke protocol

- Suite：`n_task_combinations=1`，seed `271828`
- Task：`SystemWifiTurnOn[0]`
- Goal：`Turn wifi on.`
- 1080×2400 executor action path：
  1. `open_app(settings)`；
  2. click `(407, 859)`：Network & internet；
  3. click `(280, 675)`：Internet；
  4. click `(965, 683)`：Wi-Fi switch。

| Check | Result |
| --- | --- |
| `/health` | success |
| Registry size | 116 task types |
| Task initialize | success |
| Score before | 0.0 |
| 4 executor actions | 4/4 success |
| Screenshot payload | 31,130,992 → 36,125,361 bytes；SHA256 changed |
| Score after | 1.0 |
| Tear down | success |
| Smoke elapsed time | 44.189 s |

第一次 full emulator/app setup 约需 4 分钟；正式 smoke 的 44.189 秒不含镜像构建与首次 app setup。
完整机器可读结果见 `summary.json`。截图响应约 31--36 MB，因此 Git 只保存 response size 与
SHA256，不保存像素副本。

## Setup warnings

- Contacts 自动 setup 未找到权限文案 `Don't allow`，上游逻辑记录 warning 后继续并保存 snapshot。
  后续 validation 必须单独检查 Contacts templates，不能把 setup failure 算作 policy failure。
- VLC 首先尝试的 APK 与 x86_64 ABI 不匹配，AndroidWorld 随后尝试兼容 APK并成功进入 VLC。
- headless emulator 报告 PulseAudio、X11 fallback 与软件 Vulkan warning；ADB、截图、输入和 reward
  均正常。

## Reproduction

```bash
python3 -m scripts.run_androidworld_environment_smoke \
  --base-url http://127.0.0.1:5000 \
  --task-type SystemWifiTurnOn \
  --task-index 0 \
  --suite-seed 271828 \
  --action-settle-seconds 2 \
  --output data/results/archive/androidworld_environment_smoke/summary.json
```

下一步是从 pinned 116-task registry 生成 SHA256 template partition manifest；在 manifest 提交前不
启动 validation rollout。
