# MobileWorld frozen GUI-Owl B0/B4 配对结果 v1

## 结论

用户提出的假设“B0 可能优于 B4”没有得到支持。相同冻结 GUI-Owl、MobileWorld
revision、117-task roster、50-step 上限、retry、visual-token 配置与确定性 decoding 下，
`recent-B0` 的 strict-117 分数显著低于已有 `recent-at-most-B4`：

| arm | observed | strict missing-as-zero | missing |
|---|---:|---:|---:|
| B0 | 17/94 = 0.180851 | 17/117 = 0.145299 | 23 |
| B4 | 27/114 = 0.236842 | 27/117 = 0.230769 | 3 |

strict-117 配对差值 `B0 − B4 = -0.085470`（**-8.55 pp**），20,000 次 task-level
paired bootstrap 95% CI 为 `[-0.153846, -0.017094]`；discordant outcomes 为
4 个 B0-only、14 个 B4-only、99 ties，exact two-sided McNemar `p=0.030884`。
因此在本次冻结 benchmark 合同下，实质证据指向 **B4 优于 B0**，不是 B0 优于 B4。

作为 missingness 敏感性分析，仅比较两臂都有结果的 93 个 task 时，B0/B4 success
为 17/25，差值 `-8.60 pp`，4 个 B0-only、12 个 B4-only，95% CI
`[-17.20 pp, 0]`、McNemar `p=0.076813`。方向不变，但这一口径单独看未达到 0.05；
不能隐去 B0 比 B4 多 20 个 policy-invalid missing 的事实，也不能把 observed-intersection
结果夸写为独立显著。

## Memory split

pre-execution split 上的 strict 结果同样没有出现 B0 优势：

| split | B0 | B4 | B0 − B4 | B0-only / B4-only |
|---|---:|---:|---:|---:|
| cross-app memory candidate（62） | 2 | 6 | -6.45 pp | 0 / 4 |
| single-app control（55） | 15 | 21 | -10.91 pp | 4 / 10 |

尤其是 62 个 cross-app memory candidates 中没有任何 B0-only success，反而有 4 个
B4-only success。这个结果支持“冻结 policy 在部分 MobileWorld task 上能利用历史图”
这一现象层判断；它不证明 B4 中每张图都有用，也不直接验证 learned selector 或
“不总用满 budget、不总存 recent”的方法主张。

## B0 语义审计

本次没有把 `0` 当作 unlimited/full-history sentinel：

- `select_mobileworld_memory(..., budget=0)` 对 recent/full 两种路径均显式返回空集合；
- online policy 以 `--max-history-images 0` fail closed；
- 两个 shard 合计审计 **5,455** 个 prompt，实际
  `maximum_history_images = 0`；
- 合计 5,381 个有效 policy requests、74 个 action parse failures；全部失败来自
  official-tool action 规范化/坐标约束，没有 history guard failure。

74 个 parse failures 的分类为：60 个第一坐标越界、8 个第二坐标越界、3 个 Unicode
NFKC 非规范化、2 个不支持的 `input_text`、1 个非 canonical action。它们是 B0
出现 23 个 missing 的直接执行背景；strict benchmark 按冻结合同将 missing 计 0。

## 运行时间与并发

Aries 上使用 2 × RTX A6000；每张卡一个共享 GUI-Owl policy service，各自并发驱动
8 个独立 emulator containers，冻结 roster 分为 59/58 两片。首个 runner 启动到最后
一片结束的真实 makespan 为 **7,436.473 秒（2:03:56.473）**。

已有 B4 campaign makespan 为 6:21:03.108，但它先以单 GPU 运行，之后才切换为双 GPU，
而 B0 从正式开始即近乎全程双 GPU。两者不能用来估计纯粹由 memory budget 导致的
速度倍率；B0/B4 的科学比较使用 task success，不使用 wall-time 比值。

## 冻结身份与 provenance

- scientific execution Git：
  `0efc728c8a8eb576d0ba0994600532119b0ef4e3`
- reducer Git：
  `16a27e7348e7f64056bf9f0e031af2ff2cfcc802`
- MobileWorld：
  `8ae506487bf87785292d6cad101c49955d704d39`
- environment image：
  `ghcr.io/tongyi-mai/mobile_world@sha256:b680380eac98a7ad064707f9653772af18554d201a3e6e7cf8f15d58cdc73240`
- GUI-Owl：
  `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`
- runtime：PyTorch `2.11.0+cu130`、Transformers `5.6.0`、bfloat16、
  2,560 visual tokens/image
- bootstrap：seed `20260726`，20,000 iterations

## Artifacts

- machine-readable paired result：[`summary.json`](summary.json)，SHA256
  `39ff776b745a288cbb22a382e9efd5be22bcf292b3963a17ce8aa3f4ec50de5d`
- run provenance：[`provenance.json`](provenance.json)
- B0 config：
  [`code/configs/causalcache_mobileworld_memory_b0_v1.json`](../../../code/configs/causalcache_mobileworld_memory_b0_v1.json)
- frozen shards：
  [`data/manifests/mobileworld_b0_strict117_shards_v1/`](../../manifests/mobileworld_b0_strict117_shards_v1/)
- reducer：
  [`code/scripts/reduce_mobileworld_b0_b4.py`](../../../code/scripts/reduce_mobileworld_b0_b4.py)
- B0 raw run root：Aries
  `/mnt/data6/jiaxuanluo/runs/mw-memory-b0-strict117-v2`
- B4 raw run roots：Aries
  `/mnt/data6/jiaxuanluo/runs/mw-memory-v1/{full-pinned,full-resume-gpu0,full-resume-gpu1}`

raw trajectories 状态为 `LOCAL_PRIVATE_RAW_TRACE`，不进入 Git。正式结束后两个 policy
process、nested dockerd 与 16 个 manifest 精确指定的 emulator containers 均已停止/清理；
canonical `sglang-omni-jaxan` 保留，GPU 0/1 均回到 5 MiB、0% utilization。
