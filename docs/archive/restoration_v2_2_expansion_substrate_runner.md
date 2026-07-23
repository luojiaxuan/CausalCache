# Restoration v2.2 expansion substrate runner

本文件记录 192-state label-expansion substrate 的正式 runner 执行契约。它只生成 reference substrate
测量，不生成 restoration labels，不训练 gate，也不运行 matched-NLL 或 closed-loop。

## 固定工作量

- denominator：64 trajectories、192 decision states；
- worker `even`：`cuda:0`、偶数 state index、96 states；
- worker `odd`：`cuda:1`、奇数 state index、96 states；
- 每个 state 最多执行 2 次 native generation、3 次 teacher forward、2 次 KL measurement；
- 全局计划上限为 384 / 576 / 384；不允许 retry、resume、top-up 或 replacement。

## 两阶段 source freeze

正式执行使用两个 Git commit：

1. commit A 固定 runner、artifact manager 和 parent scientific source 共 67 个文件；
2. 由 commit A 确定性生成 `code/configs/causalcache_restoration_v2_2_expansion_substrate_runner_v1.json`；
3. commit B 只把 freeze 纳入 canonical `main`，并保持 67 个执行源文件与 commit A 字节相同；
4. 执行时要求 clean `main == origin/main == commit B`，且 commit A 是 commit B 的 ancestor。

run contract 同时保存 `runner_source_git_commit` 与 `execution_git_commit`，不能用一个布尔字段替代
Git blob、remote main 和 source inventory 的实际校验。

## Preclaim 顺序

在创建全局 O_EXCL ledger 之前，runner 会完成：

1. 校验 base config、runner freeze、67-file inventory 与 clean pushed main；
2. 校验 HF derived artifact 的 exact-six clean projection、immutable revision 和 tree SHA；
3. 对 GUI-Owl 本地 snapshot 的全部 14 个文件以及 pinned Transformers source 做 file-only hash；
4. 校验 Hyper host、H200 container、两 worker topology、preflight 与 monitor-ready witness；
5. 构建 run contract，并立即交给 raw-artifact validator 做 builder-to-validator round trip。

这些步骤不构造 policy，也不执行 policy forward。

## 全局生成前 barrier

worker 构造 frozen eager runtime 后先写 runtime envelope，然后等待 coordinator：

1. coordinator exact 校验两个 envelope 的 schema、run hash、worker identity、UTC timestamp 和 runtime metadata；
2. 两张物理 GPU identity 必须不同，科学 runtime metadata 必须相同；
3. coordinator atomic 发布 runtime release；worker 会重新校验 release exact schema 和两个 runtime 文件 hash；
4. 两 worker 分别对固定 96 states 运行真实 processor canary，但不执行 policy forward；
5. coordinator 合并成全局 192-state canary，要求 192 次 included mutation、128 个 excluded states、192 次 excluded mutation；
6. archived monitor-ready 必须早于 runtime release；coordinator 才 atomic 发布 canary release；
7. worker 重新校验两个 canary report hash 和全部计数后，才能写第一个 state attempt marker。

因此任何 generation 之前都必须完成全局 192-state processor canary，而不是只做 worker-local check。

## GPU monitor sidecar

monitor 由同一个 frozen runner 的 `monitor-sidecar` 子命令生产，不能手写 ready 或 summary。sidecar 只接受
三个 absolute path；stop request path 由 summary path 确定性派生为
`<summary-file>.stop-request.json`。ready、JSONL log、summary 和 stop request 四个路径在启动时都必须
不存在，所有发布均采用 O_EXCL，旧文件不会被覆盖或复用。

在只暴露两张目标 GPU 的正式容器中，可直接复制以下命令启动并等待 ready：

```bash
set -euo pipefail
cd /data/CausalCache

MONITOR_PREFIX=/data/experiments/causalcache/.restoration-v2-2-label-expansion-substrate-v1.monitor
MONITOR_READY=${MONITOR_PREFIX}.ready.json
MONITOR_LOG=${MONITOR_PREFIX}.jsonl
MONITOR_SUMMARY=${MONITOR_PREFIX}.summary.json
MONITOR_STOP=${MONITOR_SUMMARY}.stop-request.json
MONITOR_STDOUT=${MONITOR_PREFIX}.stdout.log

for path in \
  "${MONITOR_READY}" \
  "${MONITOR_LOG}" \
  "${MONITOR_SUMMARY}" \
  "${MONITOR_STOP}" \
  "${MONITOR_STDOUT}"; do
  test ! -e "${path}"
done

PYTHONPATH=code python3 code/scripts/run_restoration_v2_2_expansion_substrate.py \
  monitor-sidecar \
  --ready-file "${MONITOR_READY}" \
  --log-file "${MONITOR_LOG}" \
  --summary-file "${MONITOR_SUMMARY}" \
  >"${MONITOR_STDOUT}" 2>&1 &
MONITOR_PID=$!

for _ in $(seq 1 120); do
  if test -s "${MONITOR_READY}"; then
    break
  fi
  kill -0 "${MONITOR_PID}"
  sleep 0.25
done
test -s "${MONITOR_READY}"
kill -0 "${MONITOR_PID}"
```

随后同一个 shell 中运行正式 runner，并把三个 producer path 原样传入：

```bash
PYTHONPATH=code python3 code/scripts/run_restoration_v2_2_expansion_substrate.py \
  <其余已冻结的正式参数> \
  --utilization-monitor-log "${MONITOR_LOG}" \
  --monitor-ready-file "${MONITOR_READY}" \
  --monitor-summary "${MONITOR_SUMMARY}"

wait "${MONITOR_PID}"
test -s "${MONITOR_SUMMARY}"
```

不要手动创建 stop request，也不要在 worker 结束时直接 kill sidecar。正式 runner 在 worker launcher 的
`finally` 中用本次 `run_contract_sha256` O_EXCL 发布 stop request；sidecar exact 校验后关闭 log，再写
summary。terminal finalizer 要求 stop 与 summary 中的 run hash 相同，且把两者的 bytes/hash 一并归档。

summary 的 exact schema 固定为 13 个字段：`schema_version`、`status`、
`run_contract_sha256`、`container_name_prefix`、`minimum_gpu_utilization_percent`、
`started_before_first_generation`、`started_at_utc`、`stopped_at_utc`、`sample_count`、
`low_utilization_incident_count`、`visible_gpu_count`、`visible_gpu_uuids` 和
`stop_request_sha256`。missing/extra key、错误 run hash、变化的两张 GPU identity 或已有的任一路径都会
fail closed。

monitor-ready 文件采用 exact schema：

```json
{
  "schema_version": "1.0.0",
  "status": "GPU_UTILIZATION_MONITOR_READY",
  "container_name_prefix": "sglang-omni-jaxan",
  "minimum_gpu_utilization_percent": 90,
  "monitor_process_alive": true,
  "monitor_pid": 12345,
  "started_at_utc": "2026-07-16T20:00:00.000000Z"
}
```

runner 会把它归档为 `logs/gpu_utilization_monitor.ready.json` 并在 terminal execution evidence 中
hash-bind。最终 monitor summary 最多等待 60 秒；缺失或不合法会形成显式 post-claim INVALID，不能冒充
preclaim failure 或科学 PASS/NO-GO。

## Crash window 与 source of truth

所有 immutable JSON（run manifest、barrier、marker、state、worker terminal、execution evidence、aggregate）
先写到 attempt root 外、同一文件系统的 sibling staging directory，完成 `fsync` 后通过 hard-link atomic
发布。SIGKILL 最多留下 root 外的 orphan temp，不会让 raw artifact collector 读到截断 final JSON。

worker sibling ledger 在每个 state 前先持久化 attempted high-water，随后 atomic 发布 marker，再更新 root
ledger。INVALID 允许保留有界 crash window，但不允许自动重跑或补齐。

代码与轻量结果进入 Git；正式 raw artifact 打包后进入 frozen HF dataset path。runner/source freeze 未完成
commit A/B 并 push 之前，本文件不构成 GPU 执行授权。

## 正式执行结果

唯一 formal attempt 已从 runner source A=`1a3833d6951c768ce1bdd5f976d1044c291d002e`、execution
B=`642feb28b4f7ce4e7bf9f7791f7fb0f6919c1839` 在 Hyper00 两张 H200 上完成。全局 processor canary
192/128/192 通过后，even/odd 各完成 96 states；192/192 valid、0 failed，384 generation、576 teacher
forward、384 KL measurement 精确命中计划。parse、finite logits、repeat agreement 均为 1.0，mean repeat
KL 为 0，185/192 states memory-sensitive，正式 outcome 为
`PASS_V2_2_LABEL_EXPANSION_SUBSTRATE_V1`。

raw 406-file deterministic USTAR 位于 private HF dataset
`gavinlaw/causalcache-restoration-v2-2-label-expansion-substrate-mobile`，tag
`v2.2-label-expansion-substrate-v1`，immutable revision
`25ac19cf6ef98adc243d421cd0039ac104ddb539`。archive SHA256 为
`4e77a38be34cb2f3c084a13abd47c0530e6729ff6cca72793977c62fa78ff47d`；fresh immutable download 已逐
byte 相同并通过同一 raw reducer。轻量结果见
`data/results/archive/restoration_v2_2_label_expansion_substrate_v1/`。

monitor 的 2,368 samples 中 803 个低于 90%；它按冻结协议从 preclaim 前开始，覆盖模型加载和明确禁止
policy forward 的 processor canary，因此不能把该全窗口计数直接解释为 state-kernel 利用率。所有 sample
原样保留；进入真实 state kernel 后，外部连续 10 秒监控多数窗口为 90%--100%。

该 PASS 只授权后续 expansion-label source/freeze 工作，不表示 restoration labels、gate、matched-NLL 或
closed-loop 已完成。

## 本地验证

```bash
PYTHONPATH=code python3 -m unittest discover -s code/tests \
  -p 'test_run_restoration_v2_2_expansion_substrate.py' -v
```

测试覆盖 parity/count contract、runtime envelope fail-closed、worker release fail-closed、root 外 atomic
publication、run-contract builder 到正式 artifact validator 的 round trip，以及 monitor producer
round trip、stop request run-hash binding、四路径 O_EXCL 与 summary exact schema。
