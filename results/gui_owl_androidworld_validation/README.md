---
pretty_name: CausalCache GUI-Owl AndroidWorld Native Validation
language:
  - zh
tags:
  - androidworld
  - gui-agent
  - policy-validation
---

# GUI-Owl AndroidWorld native-resolution validation（valid rejection）

## 结论

GUI-Owl-1.5-8B-Instruct 未通过预注册的 AndroidWorld frozen-policy gate，不能作为
CausalCache 主实验的 validated teacher。正式运行在 47/62 个原子 checkpoint 后 early-stop：
已有 15 个 official success，剩余 15 个即使全部成功也最多得到 30/62，低于 50% gate 所需的
31/62。

本次与此前 configuration-invalid 尝试不同：policy 使用 pinned adapter 的 model-default visual
preprocessing。所有 496 个 model steps 都观测到 `[1,150,68]` image grid，对应每张截图 2,550
effective visual tokens；最多保留 5 张截图。结果可用于拒绝该 policy candidate。

## 冻结结果

- plan：31 个 validation task templates、62 个动态实例，hash
  `8204e7f832f1d70becd51f299977f0e4a322a080bbfab64010345c48f901b0e8`；
- checkpoint：47；未执行：15；
- official success：15；正常终止失败：23；infrastructure exception：9；
- model steps：496；parse success：496；parse coverage：100%；
- termination：28 `policy_terminated`、10 `step_budget_exhausted`、9 `exception`；
- 成功率在固定 62 分母下的下界为 15/62，上界为 30/62；部分运行受 round-robin worker
  完成顺序影响，因此不把 15/47 报告成 benchmark success rate；
- gate、prompt、action equivalence、task split、模型权重与 native preprocessing 均未在运行中调整；
- validation test partition 保持 sealed，未用于继续选模型。

9 个 exception 包括 5 个 HTTP 500，以及 4 个 fail-closed 状态校验错误：两个 live instance
与 frozen plan 不一致，两个任务在 episode 开始前 reward 已为 1。它们按预注册规则全部保留在
62 条固定分母中，没有 retry 或删除。

## Artifact

- canonical reusable trace dataset：private Hugging Face
  `gavinlaw/causalcache-androidworld-validation-mobile@v0.1.0`
  (`3fcca45fffe9842c9fcebbf5c6c27c9540bb1515`)；47 条 trace 以单个 deterministic gzip JSONL
  shard 保存；
- `summary.json`：机器可读 early-stop gate、policy revision、visual preprocessing 与 outcome 汇总；
- Git 不重复提交 47 个逐 episode 小文件；`summary.json` 中保留原 checkpoint 名称，完整内容只在
  canonical HF shard 中保存；artifact 不包含模型参数或二进制截图；
- 复现汇总：

```bash
python -m scripts.summarize_gui_owl_androidworld_validation \
  --validation-plan configs/androidworld_validation_plan.json \
  --episodes-dir /path/to/downloaded/episode-checkpoints \
  --output results/gui_owl_androidworld_validation/summary.json \
  --minimum-parse-coverage 0.95 \
  --minimum-official-success 0.5
```

该 artifact 只回答 frozen policy 是否足够可靠；它不是 CausalCache attribution、gate 或方法效果结果。
