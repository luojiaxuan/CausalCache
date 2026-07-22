# AndroidWorld 成功轨迹采集 v1(冻结设计)

状态:`FROZEN_BEFORE_EXECUTION`。目的:为 policy 端多图 SFT 与后续成功锚定 selector 标签
采集成功轨迹。这是**数据采集**,不是评估——不与任何臂比较,不进任何裁决。

## 动机(三条证据链)

1. 上限实验 v1:记忆剂量开到 B8,closed-loop 成功率无收益(STORY_DEAD,net=0);
2. selector 侧 top-4 LoRA:macro +0.032 全部来自 B3/B4,B2/Long+ 瓶颈未动,已停;
3. 离线 oracle 恢复 headroom 始终存在(+0.375~0.47)。

结论:瓶颈在 policy 消化多图历史的能力与训练目标锚点,转向成功锚定管线:
**采成功轨迹 → policy 多图 SFT → 对新 policy 重生成成功锚定标签 → 重训 selector**。

## 采集协议

- 任务域:train-60 模板 × 3 实例(`data/manifests/androidworld_train_plan_v1.json`,
  seed 314159,records SHA `81f67f1d26da...`)。**不触碰 validation/test 切分**;
- 输入格式:`summary_B0`(纯 summary 历史,模型当前最稳的闭环格式),
  shared-early=2,与上限实验协议一致;
- 采样:`do_sample=True, temperature=0.7, top_p=0.95`,每实例 4 个 seed
  (1000+k,k=0..3),episode 级 `torch.manual_seed`;零 GPU 绑定扫描(180 组合)
  排除 12 个模板(日期嵌入 goal 漂移 7、开机即满分全局开关 3、SMS 500 1、
  500+开关污染 1),最终 **48 模板 × 3 × 4 = 576 episodes**,roster 冻结于
  `data/manifests/androidworld_success_collection_roster_v1.json`;
- parse 重试:采集允许每步最多 2 次重试(温度采样下重掷即有意义)。这是
  **collection-only** 放宽,不回写任何评估协议;v1 上限裁决不受影响;
- 截图落盘:每步 current PNG + 末步 after PNG,SFT 重渲染的原料;
- 成功判定:`score==1.0 AND policy_terminated`(与全项目一致);
- 产出预期:~20-30% 成功率 → 150-220 条成功轨迹、~4-6k 决策步。

## 实现(全部增量,不动冻结路径)

- `code/causalcache/policy/gui_owl_v2_1_sampling_runtime.py`:
  `GUIOwlV21SampledToolsRuntime` 子类,仅生成调用改采样,校验逻辑与父类一致,
  metadata 记录 do_sample/temperature/top_p;
- `run_exploratory_closed_loop_episode.py` 增量参数(默认关闭,旧行为不变):
  `save_images_dir`(截图落盘)、`sample_seed`(episode 级种子)、
  `parse_retries`(默认 0);
- `code/scripts/run_androidworld_success_collection_worker.py`:
  `--episode SEED:TASK_TYPE:TASK_INDEX` 复数、断点续跑、每局后 empty_cache。

## 算力与执行

- hyper00 + hyper01 各 6×H200(用户授权),每 GPU 3 worker(上限实验已验证密度)
  → 36 workers,720 局预计 ~4-6h;
- worker:emulator 1:1;hyper01 复用在跑的 emulator,hyper00 传镜像后新起;
- 执行纪律:零 GPU 绑定扫描(180 组合)→ 20 局 GPU smoke → 全量;host 侧
  supervisor 防外部 preflight 清理;
- 产出根:`/data02/jaxan/runs/causalcache-success-collect-v1-<rev>/`,episode
  JSON + `images/<episode>/stepNNN.png`。

## SFT 重渲染(下一阶段,另行冻结训练配置)

- 只取成功轨迹;每决策步重渲染为混合保真多图 prompt,恢复配置按步混采
  {B0, recent-2/4/8, random-K};
- 训练目标 = 该步实际执行且最终成功的 `native_output` 原文(天然语法合法,
  loss 只算输出段);
- policy LoRA:vision encoder 冻结,LM 全层;配置在训练前另行冻结;
- 硬约束:policy 一旦 SFT,现有 5,108-state restoration 标签对新 policy 全部
  作废,需按成功锚定定义重生成(GPT session 接口)。
