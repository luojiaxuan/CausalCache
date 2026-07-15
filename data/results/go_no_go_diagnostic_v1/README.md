# CausalCache go/no-go diagnostic v1

## 冻结 outcome

`INCONCLUSIVE_POSITIVE`。正证据来自 decision step 8；按照预注册边界，本结果不能输出 paper-level
`GO`、不能替代 reference-policy coverage gate、不能训练 memory gate。两个 states 是在旧 Qwen
coverage 结果中观察到 full-history executable match 后固定的，因此存在 post-selection bias。

## 关键结果

| State / budget cap | Summary KL | Recent / similarity recovery | Random recovery | Oracle / restoration recovery | Oracle coalition |
| --- | ---: | ---: | ---: | ---: | --- |
| step 4 / 512 | 0.000349 | 94.1% | 82.4% | 94.1% | `[3]` |
| step 4 / 1024 | 0.000349 | 98.9% | 94.4% | 99.6% | `[1,3]` |
| step 8 / 512 | 0.109727 | 33.2% | 36.8% | 52.6% | `[1]` |
| step 8 / 1024 | 0.109727 | 48.7% | 58.1% | 84.2% | `[1,7]` |

step 8 当前观测是 Android share sheet。recent 与 RGB-histogram similarity 都选最近的 `[6,7]`；
global oracle 和 exact budget-conditioned restoration selector 选择最早的 event 1（从 launcher 打开 BBC）
与最新的 event 7（进入 share sheet）。相对 strongest baseline 的 normalized recovery gain 在 512/1024
cap 下分别为 0.158/0.261。这个案例至少说明高保真旧事件携带了 current screenshot 与 coarse summary
没有完全表达的 policy evidence；它不是“只保留最近画面”的同义改写。

两个 states 的 repeat KL 都为 0。step 8 的 $K=16$ 五个 seeds 全部选择 `[1,7]`，minimum Spearman
为 0.964、top-budget Jaccard 与 exact-selector utility ratio 均为 1.0。step 4 的 corresponding metrics
均为 1.0。全部 selected-coalition deterministic generations 仍与 validated action executable-match：
本 pilot 看到的是 action-path distribution recovery，而不是离散 action type/scroll direction 的恢复，
因此不能声称 terminal success 已提高。

## Visual / compute accounting

- processor target：256 tokens/image；实际 grid：238 effective tokens/image；
- restored event：前后两张图，实际成本 476；512/1024 cap 分别最多恢复 1/2 events；
- 38 个 reference/coalition teacher-forced forwards + 2 个 repeat probes；
- wall time 170.30 s；模型 load 4.80 s；记录到的 teacher-forced model forward 合计 4.09 s，selected
  generation 13.43 s；peak 17.60 GiB；
- utilization monitor 在 model process 存活时采到三个 10 秒窗口均为 0%。直接原因是 full-vocabulary
  log-prob 被搬到 CPU 后逐 coalition 归约，GPU forward 是短 burst；扩展 pilot 前必须改成 GPU-side KL
  或 batch coalitions，不能按当前实现规模化。

## Provenance

- Git run commit：`2715f3141e6cbd475b6a9ab92034f24155f698a5`；
- config SHA256：`86f34b563295423536d4891c9aa0949790ac3fff30a907b9833d261deb8f13ee`；
- model：`Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；
- dataset：`gavinlaw/causalcache-guiodyssey-pilot-mobile@1de9c34ff029d4c01665cdaca74436ae24bff276`；
- shard SHA256：`54399e28df381c2114c1422dedc1aed17f1bd15b320e636155a070e181c33e72`；
- host：Hyper00 `node-radixark-16-0000`，physical GPU 0 / container `cuda:0`，NVIDIA H200；
- image：`sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
- Python 3.12.3、PyTorch 2.11.0+cu130、Transformers 5.6.0、BF16；
- UTC：2026-07-15 06:05:01--06:07:51；
- remote debug summary SHA256：`21df0507df23c7c269be30d511693824114bd3e4831e8f40ce8ebd70a9a392e0`。

`summary.json` 是 canonical compact result，`coalition_distances.csv` 保存全部 36 个可行 mixed-fidelity
coalition 的 mean/sum KL。209 KiB remote debug summary 中的 per-token arrays 与重复 runtime metadata
不作为 reusable artifact；关键 provenance、distance 与 selector result 已无损压缩到 Git。

## 下一步

按 `docs/go_no_go.md` 进入独立多轨迹阶段：在任何新 policy inference 前冻结 GUIOdyssey manifest，
保持 UI-TARS exact revision、原 50% executable-match gate、prompt/parser/equivalence 不变。gate 失败即
停止当前 reference route；通过后才运行至少 20 个未观察 restoration 的 states。不能把本目录的正案例
加入训练或用来调 threshold。
