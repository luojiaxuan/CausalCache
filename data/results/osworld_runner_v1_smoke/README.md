# OSWorld runner v1 live smoke

状态：`VALID_OSWORLD_RUNNER_LIVE_SMOKE`。

Hyper01 上使用 KVM + 官方 Docker provider 完成真实链路：

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

结果为 2 steps、3 screenshots、`COMPLETE_OSWORLD_EPISODE`，suite 返回
`COMPLETE_OSWORLD_SHARD`。同一命令第二次运行没有启动 VM，直接返回 `resumed_skips=1`，证明 task-level
completion marker 有效。

score=`0.0` 是预期值：scripted smoke 只验证 executor/evaluator plumbing，没有执行“把 Bing 设为默认搜索
引擎”的任务动作。该值不能解释为 policy 或 CausalCache 方法结果。

首次启动暴露官方 Docker image 的 dnsmasq inotify failure。v1 adapter 使用
`--no-resolv --no-poll --server=127.0.0.11` 禁止 watcher，同时保留 Docker embedded DNS upstream；未修改
host sysctl。修复后 VM server 正常通过 host port 提供 screenshot/action/evaluator 接口。

完整轻量 provenance 见 [`summary.json`](summary.json)。raw screenshots/result 留在 Hyper01 persistent path，
只作为 infrastructure smoke，不属于 reusable dataset 或 paper artifact，因此不上传 Hugging Face。
