# 全层 LoRA(无门控)v5-selected 离线 DiD 门控

## 这一轮回答什么

"能不能干脆全层 LoRA 微调 policy,不要 HGKV 这个门控探针?"——审稿人必问,也是内部一直悬着的问题。

## 配置

- adapter:`full_policy_lora`,全层 `q_proj/k_proj/v_proj`(r8/α16)。
  去掉 `o_proj` 是内存所迫:全层 q/k/v/o 在 8B 上实测差 394MB OOM,而
  `sparse_history` 明令禁用梯度检查点(重算跑在 autograd 线程,adapter 的
  ContextVar 读不到),没有别的省法。
- 语料:`desktop-did-corpus-v5-selected`(**修复版**,见下),12452 训练单元、
  205 heldout episode。
- 预算:300 optimizer steps × accum 2 × 4 ranks = 2400 单元访问 ≈ **0.19 epoch**
  (`epochs: 3` 只是上限,真正的闸门是 `max_optimizer_steps`)。
- `history_ce_weight = 0`:sparse_history 的 DiD 路径根本不消费这个参数(只有
  legacy 分支读它),设非零值会被守门拒绝且即便放行也静默失效。k=0/纯 recent
  的行为由 drift-cap(ε=0.02)锚定。

## 语料修复(重要)

盘上的 v5 b2/b4 是**修复前的构建器**(2026-07-28 20:50 版本)产出的,
`distractor_source_step` 记成了 wrong_set 的 max,常常指向 kept-recent 槽位,
触发 trainer 的 age 校验(4320 组违规)。

处理方式是**修元数据而非重建**:W 臂的渲染和图像一直是对的,真值 =
`WA.selected_steps − SA.selected_steps` 中最老者。结果:8787 组修正、5240 组
本就合规、473 组(3.3%)不可修复丢弃,k=2/3/4 多槽覆盖完整保留。
原始文件留档为 `samples_prerepair.jsonl`。

另外两条路都会毁掉 v5 的意义,故未采用:
- 直接过滤违规组 → B=4 从 21640 行砍到 1330,k=3 从 7250 塌到 135;
- 用修复版构建器重建 → 新增的证据命中过滤只剩 b2 464 组 / b4 172 组。

## 结果(两套 dev 口径一致)

| ckpt | did_select (v4b1) | recent_abs | wrong_abs | 门控 |
|---|---|---|---|---|
| step50  | +0.0003 [-0.006,+0.006] | 0.0179 | 0.0030 | ✗ |
| step100 | +0.0192 [+0.013,+0.026] | 0.0222 | 0.0107 | ✗ |
| step150 | +0.0225 [+0.014,+0.031] | 0.0230 | 0.0155 | ✗ |
| step200 | +0.0291 [+0.021,+0.038] | 0.0245 | 0.0307 | ✗ |
| step250 | +0.0279 [+0.019,+0.037] | 0.0247 | 0.0219 | ✗ |
| step300 | +0.0215 [+0.013,+0.031] | 0.0267 | 0.0262 | ✗ |

- 过的门:`did_select > 0 @ci_low`、`adapter_on_sparse > 0`;
- 不过的门:`adapter_on_recent_abs < 0.02`、`wrong_drift_abs < 0.02`。
- 没有甜点:step50 选择性还是零时,recent 漂移已 0.0179 顶线,此后单调升到
  0.0267 —— **多训不会救中性,只会更糟**,这也说明 0.19 epoch 在漂移这一维上
  不是限制因素。

## 已由对照解决(见 `../desktop_did_v5_gated36_matched`)

同语料同预算的门控 last-36 对照已跑完:门控版 did_select 峰值 +0.0145、recent
漂移 0.011–0.014、wrong 漂移 0.003–0.010,**全部过门**。因此下面这段"尚不能下的
结论"已经有了答案——漂移由**门控**控制(wrong 漂移 6 倍差),不是层范围;代价是
选择性减半。原文保留如下以记录当时的判断状态。

## 尚不能下的结论(等 gated36-matched)

不能据此说"全层导致漂移、HGKV 门控因此必要"。已有的 HGKV 报告
(`../desktop_did_v5_selected/gate_report_v5a_*.json`,last_8 与 last_36)
**训练语料只有 1078 个单元**(反复看 2.2 遍),与本轮 12452 个单元(各看一遍)
差 11.5 倍,多槽组合也只有本轮有。层范围、门控、数据多样性三个因素纠缠。

决定性对照:`gated36-matched` —— `history_gated_kv` + `last_36`,训在**同一份
修复语料、同一预算**上。它与本轮只差"有无门控"这一个因素。
