# GUI-Odyssey 训练 → AndroidWorld 零样本闭环:失败记录与诊断

## 背景:为什么做这个实验

paper 的初始方案是在 AndroidWorld 自己的 train 切分上采样、训练、评测。这个方案有软肋:
训练和评测同分布,reviewer 会质疑模板过拟合。于是改用更标准的范式:**用外部语料
GUI-Odyssey(约 900 条人类演示轨迹、201 个应用)训练,AndroidWorld 整体作为零样本测试集**。
如果成立,故事更硬:"外部数据训练、benchmark 零样本验证"。

## 做了什么

1. 把 GUI-Odyssey 轨迹渲染成与闭环推理逐字节同构的训练样本(26,195 个,含四种对照变体),
   目标动作用原始注释的精确像素坐标(换算到模型的 [0,999] 坐标系);
2. 用与 AndroidWorld 版完全相同的配方训练 LoRA(margin 对照 + CE,350 步,每 25 步存档);
3. 训练侧指标一路健康:模型区分"正确历史图 vs 错误历史图"的能力(内容敏感性)在
   AndroidWorld held-out 样本上通过统计认证(correct−shuffled +0.0070,95% CI
   [+0.0052,+0.0091];correct−irrelevant +0.0069 [+0.0047,+0.0091])——**离线指标全绿**;
4. 拿最优档(step 75)去跑真正的闭环测试:15 个记忆密集模板 × 3 实例 × 2 记忆档(无图/8 图),
   共 90 局,和冻结基线配对对比。

## 结果:两轮全败

| 条件 | 无图档成功 | 8 图档成功 | 语法崩溃局 | 主动收尾局 |
|---|---:|---:|---:|---:|
| 冻结基线 | 3/45 | 6/45 | 26/90 | 17 |
| Odyssey 第一轮(step 75) | 0/45 | 0/45 | 3/90 | **0** |
| Odyssey 第二轮(修复后 step60) | 0/45 | 0/45 | **0/90** | 15 |

**第一轮诊断**:90 局里模型一次都没有发出"任务完成"动作——不收尾就永远判失败。查训练数据:
26,195 个样本的目标动作分布是 click 6098 / swipe 1133 / type 876 / long_press 47,
**terminate 数量为 0**。原因:训练状态清单来自旧标注管线,只含轨迹中段的决策点,
从不包含最后一步。修复:为每条轨迹合成"终止态"样本(全部历史 + 最后画面 → 完成动作),
共 3,233 个,混入 25% 常规样本从 step75 续训 60 步。10 局冒烟验证:9 局里 4 局正常收尾,
修复生效。

**第二轮诊断**:语法完美(0 崩溃)、会收尾(15 局)——但 15 次收尾时任务得分全是 0
(其中 3 局第 1 步就宣布完成,其余散布在 8-31 步),没收尾的局也全部耗尽步数预算。
结论:**不是单一 bug,是人类演示的操作分布把模型的 AndroidWorld 执行力整体带偏了**
(不同应用、不同布局、不同操作习惯,135 个优化步足以造成负迁移)。

## 与成功路线的对照

同样的配方、同样的评测,用 AndroidWorld 自采成功轨迹训练的模型(v3-e1):
无图档 3/45 → **7/45**,配对提升 +8.9 个百分点,95% CI [+0.022,+0.156],统计认证通过。
差别只有训练数据来源——**同分布的自采数据能提升闭环,外部人类演示数据反而破坏闭环**。

## 对 paper 的含义

1. 零样本内容敏感性认证(离线)是真实的正结果:外部数据 75 步就能教会模型"读历史图内容",
   且跨语料迁移;
2. 闭环 0/90 与离线全绿并存,是全文主题(离线指标与闭环成功的层层脱节)的最强实例:
   teacher-forced 能力 ≠ 自由生成格式 ≠ 闭环任务能力,三层各自独立;
3. 主结果采用 AndroidWorld 自训模型;Odyssey 线作为迁移分析章(正的离线迁移 + 负的闭环迁移
   + 两层根因),不作为主方法。

## 数据位置

- 第一轮 90 局:hyper01 `/data02/jaxan/runs/causalcache-assay-zeroshot-v1/episodes/`
- 第二轮 90 局:hyper01 `/data02/jaxan/runs/causalcache-assay-zeroshot-v2/episodes/`
  (H100 `/data/jaxan/causalcache/runs/zshot2/`、hyper00 `/data02/jaxan/runs/zshot2/` 为冗余副本)
- 训练 checkpoints:hyper00 `/data02/jaxan/runs/causalcache-odyssey-margin-v1{,b}/`
- 训练数据集:hyper00 `/data02/jaxan/artifacts/sft/causalcache-odyssey-sft-v1/`(动作分布见上)
- 全部 `PENDING_HF_UPLOAD`

## 更正(2026-07-23,复核后)

**上文"人类演示分布负迁移"的归因撤回**。GPT session 复核仓库后指出、且数据验尸确认了两个工程 bug:

1. **坐标被二次归一化**:GUIOdyssey 官方注释坐标本就归一化到 [0,1000)(400 条轨迹抽样 9,378 个
   坐标点,0 个超过 1000,max=999、p95≈930——若是绝对像素,3120 高的屏上 y 的 p95 应达 ~2800)。
   渲染器又除了一次设备分辨率,所有 click/swipe/long_press 目标被系统性拽向屏幕左上。
   这解释了"语法完美、会点击、但永远点错位置"的全部闭环现象;
2. **CLICK+KEY_\* 字符串动作被静默丢弃**(抽样中 KEY_HOME 498 / KEY_BACK 14 / KEY_APPSELECT 14,
   约占动作 9%)——训练目标零 system_button 的直接原因;另有 ~4% INCOMPLETE 轨迹被硬标 success 混入。

**因此前两轮 0/90 定性为 invalidated engineering runs,不构成 negative transfer 的科学证据**。
已修复(坐标直映射 [0,999]、KEY_HOME/BACK→system_button、INCOMPLETE 轨迹排除),按限额 salvage
方案重渲染重训(≤75 步、terminal 自然配比、四层 gate + 10 局 canary 后才准 90 局);paper 主线
(v3-e1 + selector)不等待该支线。终止态过采样(前轮续训 terminal 占 1/3)的教训一并吸收。

## Salvage 轮 canary 读数(2026-07-23,坐标+动作修复后;裁决被下方 90 局推翻)

- 修复后数据集(37,635 样本):动作分布健康(system_button 979、terminate 1000 自然配比);
- **离线四层 gate 首次全过**:历史效用 correct−b0 = +0.010~+0.012(CI 全正,坏坐标时代为负)、
  内容对照 c−irrel CI 正、AW 迁移 c−irrel CI 正——坐标修复直接改变 gate 符号;
- 闭环 canary(12 局):6 局环境故障(新启 emulator 未稳期瞬态),有效 6 局 0 成功、3/6 提前
  终止——当时据此记录"线收档"。
- **和解注记(2026-07-24)**:上述收档裁决基于 12 局小样本,为时过早。用户过夜指令下的
  90 局全量化验尺(emulator 稳定后)推翻了"系统性提前终止"的判读:52/90 正常终止、
  B0 成功翻倍、语法崩溃 26→1。canary 的 0/6 属小样本 + 未稳环境噪声。收档结论作废,
  见下节 90 局裁决为准;canary 教训(新 emulator 需稳定期)已入 progress.md。

## 第三轮(2026-07-23 夜,坐标修复版 s75):零样本泛化成立

| | frozen | odyv2-s75 |
|---|---:|---:|
| B0 / B8 | 3/45 / 6/45 | **6/45** / 6/45 |
| parse 死亡 | 26/90 | **1/90** |
| policy_terminated | 17 | **52** |
| 配对 s75-B0 vs frozen-B0 | — | **净胜 +3(3W 0L)Δ+0.067 CI[+0.000,+0.133]** |
| s75-B8 vs frozen-B8 / 内部 B8−B0 | — | 平(net 0) |

结论:**坐标修复后,纯 benchmark-external 的 GUI-Odyssey 训练零样本迁移到 AndroidWorld 成立**
(基础能力层:B0 成功率翻倍、零模板负局、语法崩溃 26→1、终止行为正常)。与 AW 自训 v3-e1
(+8.9pt CI 认证)同方向、略弱——外部数据代价可接受。记忆剂量收益(B8−B0)仍平,与全线一致:
"恢复历史→成功率"的最后一跳留待 selector 链/history-gated 线解决。前两轮 0/90 的
invalidated-runs 定性维持;本轮为有效的 zero-shot training transfer 数据点(checkpoint 选择
用过 AW heldout 信号,措辞按此收窄;终测另封 sealed 模板)。
- 数据:hyper01 /data02/jaxan/runs/odyv2-zeroshot-full/(90 局,infra 0 / exec 4),PENDING_HF_UPLOAD。

