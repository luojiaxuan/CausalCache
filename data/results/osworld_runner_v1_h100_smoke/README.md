# OSWorld runner v1 H100 portability smoke

状态：`VALID_OSWORLD_RUNNER_H100_LIVE_SMOKE`。

H100 上 pinned OSWorld preflight 返回 10 domains / 369 tasks。默认 official Docker runtime 使用
`CPU_MODEL=host` 时，KVM QEMU 已启动但 Ubuntu 停在 GRUB/early boot，300 秒内 screenshot endpoint 未 ready。
同一 OSWorld revision、qcow2 和 VM Docker digest 只改为 `CPU_MODEL=qemu64` 后约 12 秒 ready。

正式 runner 随后完成：

```text
DesktopEnv init
-> reset(public Chrome task)
-> screenshot
-> WAIT
-> screenshot
-> DONE
-> screenshot
-> evaluate
-> close/remove VM container
```

结果为 2 steps、3 screenshots、`COMPLETE_OSWORLD_EPISODE`；相同命令第二次返回 `resumed_skips=1`，且没有
残留 OSWorld container。score=`0.0` 是 scripted executor smoke 的预期值，不是 policy performance。

H100 profile 使用显式 `--docker-cpu-model qemu64`，该值已写入 episode provenance。这个 workload 只使用
CPU/KVM，没有分配 H100 GPU。raw screenshots/result 保留在 H100 persistent path；它只是 infrastructure smoke，
不是 reusable dataset 或 paper artifact，因此不上传 Hugging Face。完整轻量 provenance 见
[`summary.json`](summary.json)。
