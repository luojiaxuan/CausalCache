# Qwen3-VL full-history executable-match coverage

## 结论

`Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` **不适合作为当前 CausalCache 主实验的 frozen reference policy**。

在成功 GUIOdyssey trajectory `0054832199799795` 的全部 9 个 decision 上，full-history generation 只有 1 个 action 与 recorded action executable-match，coverage 为 11.1%。唯一通过的是 `type_text`；所有 tap 和 swipe 均未通过。

| Action type | Decisions | Parseable | Matches | Coverage |
| --- | ---: | ---: | ---: | ---: |
| tap | 7 | 4 | 0 | 0% |
| swipe | 1 | 1 | 0 | 0% |
| type_text | 1 | 1 | 1 | 100% |
| overall | 9 | 6 | 1 | 11.1% |

3 个长历史状态输出了 `target: coordinate_bin:*`，而不是约定的可执行 coordinate，因此按预注册 contract 记为 parse failure。其余视觉动作虽然可解析，但 target coordinate bin 不匹配。不能通过放宽验证标准把这些状态加入 attribution 数据。

该结果只拒绝此通用 backbone 作为主 teacher，不是否定 CausalCache 方法。完整记录见 [`summary.json`](summary.json)。

## 资源范围

- Visual tokens：每张图 256
- 最大 full-history input：4,041 tokens
- 最大 peak allocated GPU memory：18.84 GB
- Runtime：单张 NVIDIA RTX A6000、bfloat16、PyTorch 2.11.0+cu130、Transformers 5.6.0

## 复现命令

```bash
python3 -m scripts.run_qwen_policy_coverage \
  --dataset-tar /data/artifacts/causalcache-guiodyssey-pilot-mobile/data/guiodyssey-pilot-00000.tar \
  --model-dir /data/artifacts/models/Qwen3-VL-8B-Instruct \
  --device cuda:0 \
  --visual-tokens-per-image 256 \
  --max-new-tokens 96 \
  --output-dir /data/experiments/qwen_policy_coverage_pilot
```
