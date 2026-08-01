# B 预算扫描 v1:设计缺陷导致实验无效,已作废

## 结论先行

**这轮扫描回答不了"最优 B 是多少"。六个 recent 臂实际全是 Recent-4 跑了六遍。**
已在第 7/11 个配置时终止,机时约 10 小时。数据保留作为复现噪声样本,不得用于预算结论。

## 缺陷

预算 `$b` 从未到达真正决定记忆分配的地方。

1. 驱动对**所有**配置传同一个固定 config:
   `causalcache_mobileworld_official_b4_hgkv_sel_v1.json`,其中写死
   `memory_arm: "full"`、`memory_budget: 4`;
2. `run_mobileworld_gui_owl.py` 的预算取自该 config(`config["policy"]["memory_budget"]`),
   **不看扫描变量**;
3. `$b` 只传给 `bsweep_serve.sh`,而且只在 `sel` 臂被使用
   (`--selection-budget "$BUD"`);**recent 臂收到 `$b` 后完全没有使用**。

于是:

- **recent 臂**:B=0/1/2/4 四个配置的逐任务结果**字节级完全相同**(36/36),
  成功任务名单一字不差 —— 它们都是同一个 Recent-4;
- **sel 臂**:selector 被告知选 B 个,但 runner 仍按 4 个槽位渲染,语义不一致,
  不能解释为"预算 B 下的 selector"。

## 是怎么发现的

不是靠读代码,是靠一个**反常的巧合**:归约时发现 recent 臂在 B=0/1/2/4 上成功率
全部恰好等于 25.00%(9/36)。四个不同预算给出同一个数字,概率极低,于是做了逐任务
比对,36/36 一致 —— 立刻坐实。

**教训:多臂实验归约时,先做一次"各臂结果是否真的不同"的健全性检查。**
相同的汇总数字可能是巧合,逐任务字节一致不可能是巧合。这个检查成本几乎为零,
却能在烧掉几十小时机时之前发现"参数没有真正生效"这类缺陷。

## 修法(未执行,待决定是否重跑)

runner 没有预算的命令行覆盖项,必须为每个 B **生成独立 config**,且注意
`run_mobileworld_gui_owl.py:210` 的前置校验:`last_image` 必须等于
`memory_budget + 1`,两个字段要同步改。config 属于科学参数,需入 Git。

同时应修 `bsweep_serve.sh`:recent 臂也要把预算传给服务器,而不是只给 sel 臂。

## 是否值得重跑

补充材料 Table S12 已有**全量 117 任务**的预算表(B=0/1/2/4/8,recent 与
selector 两臂),全量口径下峰值为 B=2(总体 +6.8pp)与 B=4(memory-critical +8.6pp)。
本轮 36 任务子集即便修好重跑,统计力也远不如它:子集里 control 组只有 12 个任务,
**单个任务翻转即 8.3pp**,分层结论无法解析。

因此重跑的增量价值主要是"补 B=6 这一点"与"统一 recent 臂口径",需权衡约 13 小时机时。

## 保留的数据

`/data/mw/runs/bsweep/`(hyper00 容器内),7 个配置的原始轨迹与 result.txt。
其中四个 recent 臂互为完全重复,可作为**同机同配置的复现噪声样本**——
它们逐任务一致,恰好说明确定性解码 + 同一 harness 下 MobileWorld 是可复现的,
与 OSWorld 7-9% 的翻转率形成对照(差别在于 OSWorld 有真实 VM 的时序随机性)。
