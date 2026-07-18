# AndroidWorld long-horizon incidence v1

这是对两个已经发布、early-stopped 的 AndroidWorld validation trace artifacts 所做的只读统计，目的只是
检查 `n=8/16` 是否会在当前真实执行 stack 中出现。它不是新的 closed-loop run，也不能外推成完整
AndroidWorld benchmark 的 episode-length 分布。

候选历史数量按项目当前定义计算：对 zero-based decision index `i`，
`n=max(0,i-1)`；最近一个 event 的 post-state 与 current observation 重复，因此不进入可恢复候选集。

| stack | decisions | `n>=8` decisions | nonempty episodes reaching `n>=8` | `n>=16` decisions | nonempty episodes reaching `n>=16` |
| --- | ---: | ---: | ---: | ---: | ---: |
| GUI-Owl-1.5-8B-Instruct | 496 | 179 / 36.09% | 27/38 | 64 / 12.90% | 8/38 |
| GUI-Owl-1.5-8B-Think | 513 | 257 / 50.10% | 21/33 | 160 / 31.19% | 8/33 |

Source of truth：

- private HF dataset `gavinlaw/causalcache-androidworld-validation-mobile@3fcca45fffe9842c9fcebbf5c6c27c9540bb1515`
  (`v0.1.0`, Instruct)，trace SHA256
  `dcd61490febf65cc8ecf7cfbefea371f5c8a501f4e014a0a362e8b8431b44f90`；
- 同一 private HF dataset `@0faf767e7c1f64b5f39fde1ac6913ca93337d8f2`
  (`v0.2.0`, Think)，trace SHA256
  `a5e9e771fde087f0684b016a884e37ec4300066c4b3a37e42e79303cd3e787fb`；
- 聚合结果见 [`summary.json`](summary.json)，原始 episode traces 不复制进 Git。

复现入口：

```bash
PYTHONPATH=code python3 code/scripts/run_long_horizon_incidence.py \
  --trace-shard <downloaded-trace.jsonl.gz> \
  --artifact-label <label> \
  --expected-sha256 <immutable-sha256> \
  --expected-record-count <count> \
  --output <temporary-output.json>
```

正式 long-history method evaluation 另立 `n=8,B=2/4` exact analysis 与 `n=16,B=2/4`
pair-union development protocol；本统计不授权打开原先锁定的 confirm、matched-NLL、closed-loop 或 sealed test。
