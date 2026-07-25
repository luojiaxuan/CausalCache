# 官方 GUI-Owl B=4 复现记录(封存)

## 结论

单次 rollout 下,我们的复现值与官方单次值差约 17pt;**论文报告的 69.0 是 Pass@3,不是单次成绩**。

| 条件 | 口径 | 成功率 |
|---|---|---:|
| 我们的 roster(116 局,infra 已补跑) | 单次,剔 infra | **38.0%**(35/92) |
| 官方 seed=30 实例(115 局) | 单次,剔 infra | **38.9%**(37/95) |
| MobileForge 公开 rollout | **单次** | **56.0%**(65/116) |
| 论文报告 | **Pass@2** | 68.1% |
| 论文报告 | **Pass@3** | 69.0% |

**任务实例不是差距来源**:换成官方 seed=30 的同一批实例(116 个里 79 个 goal 不同),成绩几乎不变(38.9% vs 38.0%)。

## 逐项排除的因素(对照官方源码 X-PLUG/MobileAgent @ Mobile-Agent-v3.5)

复现用的是**单体 `gui_owl.py` wrapper**,不是 `mobile_agent_v3.py` 多智能体系统;每个环境步严格一次贪心生成,无 planner/reflector/best-of-N。

| 因素 | 官方 | 我们(修复后) |
|---|---|---|
| 输出格式 | `Action:` + `<tool_call>` | 已复刻(原先禁止 Action 行) |
| 历史表征 | 模型自写的 action 文本 | 已复刻 `Previous actions: StepN:` |
| 解析 | 宽松 split,失败转 UNKNOWN 消耗一步 | 已复刻,UNKNOWN 率 4.9% |
| 视觉预算 | `last_image=5` | 映射为 `last_image = B+1` |
| 步数预算 | `int(10 × complexity)` | **本就一致** |
| 坐标 | `src_format="qwen-vl"` = `x/999×width` | **本就一致** |
| 任务实例 | `task_random_seed=30`, `n_task_combinations=1` | 已复刻,无差异 |

我们这边另修了两处自有偏差:`wait`/`long_press` 的 `time` 参数缺失(曾导致 33% 的步骤被误判 UNKNOWN)、以及"起始分必须为 0"的前置检查(官方 `suite_utils.run_task` 无此检查)。

## 残余差距的可能来源(未排除)

1. **执行环境**:我们的 emulator 有 17-21% 的局死于 infra 故障(HTTP 500、连接中断、状态污染),存活局也可能跑在退化环境里;
2. 硬件/推理栈差异(我们本地 transformers 贪心解码 vs 官方 vLLM 服务)。

## 对论文的处理

主结果一律写 **"Frozen GUI-Owl under our controlled harness"**,不拿 69.0 当我们的基线;附录记录本复现与 open discrepancy。**所有相对结论(记忆预算消融、sparse vs recent)均为同 harness、同实例、配对比较,不受绝对水平影响。**

## 数据

- 我们的 roster:hyper00+hyper01 `/data02/jaxan/runs/official-b4/`(116 局)
- 官方 seed30:hyper01 `/data02/jaxan/runs/official-b4-seed30/`(115 局)
- roster:`/data02/jaxan/runs/seed30_today.json`(seed 30, 1 combination, 116 模板)
- 全部 PENDING_HF_UPLOAD
