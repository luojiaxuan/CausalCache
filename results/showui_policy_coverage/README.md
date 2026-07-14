# ShowUI-2B full-history coverage

## 结论

固定的 `showlab/ShowUI-2B@cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60`
在预注册的 9-decision GUIOdyssey pilot gate 上得到 9/9 parsed、2/9 executable
match（22.2%），未通过至少 50% overall coverage 与每个现有 action type 至少命中一次
的双重门槛。因此 ShowUI-2B 不进入当前 attribution pilot。

## 分层结果

| Action type | Decisions | Parsed | Matches | Coverage |
| --- | ---: | ---: | ---: | ---: |
| tap | 7 | 7 | 1 | 14.3% |
| swipe | 1 | 1 | 0 | 0.0% |
| type_text | 1 | 1 | 1 | 100.0% |
| overall | 9 | 9 | 2 | 22.2% |

所有 3--19-image full-history 输入都完成 generation；最长输入为 5,305 tokens，峰值
allocated GPU memory 为 5.01 GiB。失败不是 OOM 或 parser failure：模型在 decision
steps 6、7、8、10 过早输出 `ANSWER('task complete')`，并且没有命中唯一的 swipe。
这表明当前 candidate 对该跨 app 长轨迹的 next-action behavior coverage 不足。

这是 frozen policy selection 的负结果，不是对 CausalCache restoration mechanism 的
falsification。按 gate 前登记的停止规则，不再针对该 trajectory 调 prompt、扩大坐标
容差或继续枚举相似 backbone；下一步改为选择与 benchmark 原生适配、可合法访问 logits
且具有独立 held-out validation 的 policy/evaluation stack。

## 可复现命令

```bash
python3 -m scripts.run_showui_policy_coverage \
  --dataset-tar /data/artifacts/causalcache-guiodyssey-pilot-mobile-v03/data/guiodyssey-pilot-00000.tar \
  --model-dir /data/artifacts/models/ShowUI-2B \
  --coverage-gate configs/policy_coverage_gate.json \
  --device cuda:0 \
  --visual-tokens-per-image 256 \
  --max-new-tokens 128 \
  --output-dir /data/experiments/showui_policy_coverage
```

完整逐 decision 输出、snapshot hash、latency 与显存见 [`summary.json`](summary.json)。
