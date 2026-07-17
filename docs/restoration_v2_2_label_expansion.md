# Restoration v2.2 label expansion

> 当前状态：48/16 policy-blind structural manifest 已从 clean pushed `main@d2a4705` 物化并通过 committed
> validator，SHA256 为 `4aec4deffc6405c3d06ca3001d082e4fbd85ee44f55785708a0cee573edcd169`。policy-blind
> derived artifact 已绑定 immutable revision `630363a6...`；192-state substrate 已完成唯一双 H200 attempt，
> 192/192 valid、185 memory-sensitive，正式 `PASS_V2_2_LABEL_EXPANSION_SUBSTRATE_V1`，raw artifact 绑定
> private HF immutable revision `25ac19cf6ef98adc243d421cd0039ac104ddb539`。expansion exact-label source、
> config、双 H200 runner 与 raw-artifact reducer 已进入 source A；config SHA256 为
> `65f7fa1d35a0b1fdd4fa09fe09120e858252406a3b850d85d9415ab34d6feed5`。单独的 committed/pushed runner
> freeze B 完成前仍不授权 GPU。尚未生成 expansion restoration labels、gate checkpoint 或 development
> metric；旧 20 条 confirm 继续封存。

## 为什么不能直接训练

现有 exact-label artifact 含 10 条 train 与 5 条 development trajectory。虽然它提供 420 个
$D(S)$ rows 和 435 条 conditional edges，但 edge 共享 trajectory、instruction、截图和 coalition table，
不能当成数百个独立样本。它足以支持 oracle 与 policy-free method shaping，不足以支撑 learned gate 的
跨 trajectory 泛化 claim。

因此现有角色边界改为：

- 旧 10 train：只允许 trainer/algebra smoke，不进入当前 source milestone 的新输出；
- 旧 5 development：已经用于 method shaping，不能再选择模型、epoch、threshold 或 loss；
- 新 48 train + 16 fresh development：正式 gate 数据扩展；
- 旧 20 confirm：只有在 gate、checkpoint、threshold、matched-NLL 和 confirm runner 全部冻结后才允许打开。

最终 planned denominator 是 58 train、21 development、20 confirm trajectories。正式 GO/NO-GO 以新鲜的
16 条 development 为主；旧 5 条只作 sensitivity，combined 21 只作 secondary report。

## 冻结选择规则

machine-readable config 是
[`../code/configs/causalcache_restoration_v2_2_label_expansion_v1.json`](../code/configs/causalcache_restoration_v2_2_label_expansion_v1.json)。
它只从 immutable parent selection 的 111 条 eligible records 重建：

```text
111 eligible
  - 8 reference
  - 10 legacy train
  - 5 legacy development
= 88 remaining
  -> decision_count >= 5: 84 structural
  -> assert frozen confirm == structural[:20]
  -> expansion = structural[20:]: 64
  -> first 48 train, last 16 fresh development
```

不能按 app、action type、memory sensitivity、oracle recovery、policy score 或人工观察 top-up。split 的
canonical ID-list SHA256 为：

- all 64：`238a5608d08f6e4594195690b7595b1953e0865da2192e1e4b46d8b5795b3526`；
- train 48：`11904e1d102b69561b402bea4e42044be9572884f6168597c5f07c3ca346cb34`；
- fresh development 16：`1c37cbf6b67b0ddee61b3efe27d33b30c8fbfb471ced12f12e49624a10b82454`。

结构 manifest 只输出 source/transport/selection/decision-count provenance 和 state skeleton，不输出 instruction、
action、OCR、image identity、旧 state record 或既有 role 的 source IDs。

## 固定 label geometry

每条 expansion trajectory 固定 decision steps 4、5、6，对应 candidate counts 2、3、4。完整 power set 与
部署可达 $B=2$ conditional edge 数为：

| Step | Candidates | $D(S)$ rows | Deployment edges |
| --- | ---: | ---: | ---: |
| 4 | 2 | 4 | 4 |
| 5 | 3 | 8 | 9 |
| 6 | 4 | 16 | 16 |

因此新增 workload 为：

| Role | Trajectories | States | $D(S)$ rows | Conditional edges | Teacher forwards |
| --- | ---: | ---: | ---: | ---: | ---: |
| train expansion | 48 | 144 | 1,344 | 1,392 | 1,488 |
| development expansion | 16 | 48 | 448 | 464 | 496 |
| total | 64 | 192 | 1,792 | 1,856 | 1,984 |

这里的 1,984 次 teacher forward 包含每 state 的 full-history reference、repeat reference 与除 full coalition
外的 mixed-fidelity forwards；1,792 个 KL 包含 repeat-reference KL 和所有非-full coalition KL。后续 raw
$D(S)$ 仍是唯一 canonical truth，oracle、marginal、interaction 与 attribution 全部 policy-free 重算。

## Source 与 artifact 边界

旧 derived HF revision `89f136ab...` 只覆盖旧 15 train/development 和 20 confirm，与新 64 条交集为 0，
不能作为 expansion input。新 artifact 必须从 pinned GUIOdyssey 16 个 Parquet 重新构建，并在 materialization
时重新验证 2,252,923,738 bytes 的逐文件 size/SHA256。Hyper00 当前持久盘已有一份 16/16 hash-matched source
projection，但本地盘只作 staging；新 reusable derived artifact 仍要上传 private Hugging Face、取得 immutable
revision，并 fresh-download replay。

扩展 substrate 前的五项前置状态为：

1. 从 clean pushed `main` 物化 structural split manifest（已完成）；
2. expansion exposure ledger 机械证明 64 IDs 与所有既有 policy/restoration output source union 交集为空（已完成）；
3. 只含 expansion 64 的 policy-blind derived artifact（已完成并 fresh immutable replay）；
4. gate method family、feature、loss、OOF、tie/stop 与 fresh-16 development gate（已冻结）；
5. expanded substrate（已 PASS 并闭合 immutable artifact）；expanded-label source contract 与 runner 已实现，
   仍需 source A commit/push 后单独物化、commit、push runner freeze B。

expanded-label 科学边界与唯一 attempt 见
[`restoration_v2_2_expansion_exact_labels.md`](restoration_v2_2_expansion_exact_labels.md)。当前 source-only
validator 明确不授权 label/GPU execution；只有 committed runner freeze B 的 validator 才能授权唯一双 H200
attempt。formal gate training、development tuning、matched-NLL、closed-loop、confirm 与 AndroidWorld sealed
test access 仍全部 locked。

## 验证入口

source-focused tests：

```bash
cd code
python3 -m unittest tests.test_restoration_v2_2_label_expansion -v
```

expanded exact-label source 在 source A 中额外运行：

```bash
cd code
python3 -m unittest \
  tests.test_restoration_v2_2_expansion_label_inputs \
  tests.test_restoration_v2_2_expansion_labels_contract \
  tests.test_restoration_v2_2_expansion_labels_artifact \
  tests.test_run_restoration_v2_2_expansion_labels -v
```

科学 config 固定为
[`../code/configs/causalcache_restoration_v2_2_expansion_labels_v1.json`](../code/configs/causalcache_restoration_v2_2_expansion_labels_v1.json)，
SHA256
`65f7fa1d35a0b1fdd4fa09fe09120e858252406a3b850d85d9415ab34d6feed5`。它固定 64 trajectories / 192
states、$n=2/3/4$、$B=2$、1,792 raw $D(S)$ rows、1,856 deployment edges、3,072 full edges、1,984
interactions、576 attributions、192 exact oracles、1,984 teacher forwards 与 1,792 KL；generation、expert、
gate、matched-NLL、closed-loop 和 confirm operation 均为 0。raw $D(S)$ 是唯一 canonical truth，全部 derived
labels 必须 policy-free 重算并保留 negative marginals。

结构 manifest 必须在 source commit 已 push 后从 clean canonical `main` 生成：

```bash
cd code
python3 -m scripts.materialize_restoration_v2_2_label_expansion \
  --config configs/causalcache_restoration_v2_2_label_expansion_v1.json \
  --parent-selection ../data/manifests/restoration_v2_selection.json \
  --git-revision <CLEAN_PUSHED_MAIN_SHA> \
  --output ../data/manifests/restoration_v2_2_label_expansion_selection.json
```

结果 commit 后使用 `scripts.validate_restoration_v2_2_label_expansion` 对 config、parent、generator commit 和三份
source bytes 做 committed replay。
