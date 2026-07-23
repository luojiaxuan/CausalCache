# AndroidWorld memory ceiling v1(dev,预注册)

状态:`FROZEN_BEFORE_EXECUTION`。经用户批准的上限实验:在记忆密集型 dev 模板上,用剂量阶梯回答
"给冻结策略恢复更多高保真历史,closed-loop 成功率是否上升"。这是 closed-loop 记忆主张的裁决实验,
先于任何 selector 臂。

- **臂**:`summary_B0 / recent_B2 / recent_B4 / recent_B8`(B8 为 32k 上下文下的确定性可行上限,
  逐决策记录实际恢复张数与 prompt tokens);
- **模板**:15 个,按纯先验规则从 dev-60 选出(≥24 步,或 ≥18 步且命中多条目/跨 app 转录/状态化
  遍历;排除 Retro 与全局开关类)。dev-60 模板此前从未被本项目执行过,不存在结果污染;
- **实例**:每模板 task_index 0/1/2(train plan,suite seed 314159,plan 与 records SHA 绑定于
  [config](../../code/configs/causalcache_androidworld_memory_ceiling_v1.json));共 180 episodes;
- **协议修改**:共享 summary-only 早期段从 5 步缩至 2 步;episode 间清空 GPU 缓存;
- **判定(预注册)**:B8−B0 模板级净胜 ≥ +3(或 bootstrap CI 不含 0)→ 记忆故事成立,进入 selector
  臂阶段并冻结本组合;净胜 ≤ +1 → 该 backbone 的 closed-loop 撑不起记忆主张,决策转向换 backbone 或
  重构叙事;跑完后禁止换模板/改预算/改共享段/改主指标;
- **边界**:dev 数据,不进 paper 主表;sealed test 不访问。

执行:H100,4 emulator × 4 GPU workers,runner 为
[`run_androidworld_memory_ceiling_worker.py`](../../code/scripts/run_androidworld_memory_ceiling_worker.py)
(episode 引擎扩展:ceiling 臂、plan 模式、共享段参数,validation12 旧路径行为不变)。
