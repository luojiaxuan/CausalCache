# 部署路线决策 v2:两遍 vs 单遍(2026-07-28,与同事讨论用)

前一版([method-mainline-vs-singlepass.md](method-mainline-vs-singlepass.md))
写于 mobile 单遍闭环出结果之前;本版含全部三平台终值,数据口径以本版为准。

## 0. 全景终表(closed-loop)

### MobileWorld(117 任务,零样本迁移,全部满员完赛)

| 臂 | 成功率 | vs 冻结 B0(三轮 28.21%) |
|---|---|---|
| 冻结 B0(3 轮:33/32/34) | 28.21% | — |
| HGKV recent-B4 | 29.06% | +0.85pp(n.s.) |
| **HGKV+selector 两遍**(2 轮:39/42) | **34.62%** | **+6.41pp CI[0.00,12.82] p=0.024 ✓** |
| HGKV+selector 单遍 last-action | 29.06% | +0.85pp CI[−6.55,+7.98](**失效**) |
| 单遍 vs 两遍直接配对 | — | **−5.56pp CI[−11.11,−0.43](单遍显著更差)** |

### OSWorld-Verified(361 任务,desktop=训练平台;sel1p/recent 358/361 待清扫)

| 臂 | 成功率 | 备注 |
|---|---|---|
| 冻结 B0 | **19.94%**(72/361,满员) | 多步工具域几乎全灭 |
| HGKV recent-B4 | 33.0%(118/358) | 记忆本身 +13pp |
| **HGKV+selector 单遍** | **34.4%**(123/358) | vs recent +1.3pp(配对检验待满员;预注册 steps≥6 分层待出) |

### Desktop 离线(dev 集,episode-cluster CI,全预算扫描)

| selector−Recent-B | B=1 | B=2 | B=4 | B=8 |
|---|---|---|---|---|
| 两遍(gold witness 上界) | +0.044 ✓ | +0.025 ✓ | +0.011 ✓ | ~0 |
| **单遍 last-action** | +0.039 ✓ | +0.025 ✓ | +0.015 ✓ | ~0 |
| 单遍 intent(指令) | — | +0.015 边缘 | +0.005 ✗ | — |
| 单遍无 witness | — | +0.016 ✓ | +0.009 ✗ | — |

## 1. 路线 A:两遍 propose-then-select 为主线

**证据面**:唯一在零样本 mobile 上显著超冻结基线的臂(+6.41pp,p=0.024);
desktop 离线是上界。**机制自洽**:proposal 参照每步由当前平台 policy 自我校准,
天然可迁移;这正是四点参照阶梯(gold/proposal > last-action > intent > none)的
预测。

**成本(RTF,按你的定义:额外耗时 ÷ 任务总耗时)**:
- 实测(mobile r2 全程 2224 步审计):pass-2 触发率 81.7%,中位 pass2 3.26s、
  selector 39ms → **每步额外 ≈ 2.7s**;
- mobile 每任务 ≈ 19 步 → 额外 ≈ 51s/任务;任务墙钟 4-7 分钟 →
  **RTF ≈ 12-20%**;
- OSWorld 口径(若在 desktop 也跑两遍):中位任务 249s、15 步 → 额外 ≈ 40s,
  RTF ≈ 16%;
- 注意区分:+66% 是"占 policy 计算时间"的比例;摊到含环境执行的任务总时长
  就是上面 12-20%。**你的直觉正确:RTF 不高,可向 reviewer 解释**,还可补一句
  "gap-fold 重组只在换帧时发生,且第二遍与第一遍同预算(B 张图)"。

**风险**:reviewer 仍可能嫌两遍不优雅;需要 RTF 表格 + 触发率分布图来防守。

## 2. 路线 B:单遍 last-action 为主线

**证据面**:desktop 域内足够好(离线全预算显著、OSWorld 闭环 +1.3pp 于 recent
——显著性待满员配对与 steps≥6 分层);**mobile 零样本失效是硬伤**。

**若为救 mobile 加 mobile GUI 训练数据**:
- 会破坏"selector 纯 desktop 训练、零样本迁移"的故事——这是当前 novelty 的
  重要一半(HGKV 的 policy-preserving + selector 的跨平台);
- 且不保证成功:失效点可能在 last-action 参照本身(mobile 轨迹更长、动作更
  嘈杂,上一步动作常是 scroll/输入噪声),不是特征分布;
- 工期:mobile 标签生产 + 重训 + 重跑闭环 ≈ 2-3 天,截稿前高风险。

**优点**:域内部署故事最干净(~1% 开销),解释成本最低。

## 3. 路线 C(建议):一个方法、一个部署旋钮——"域内单遍,跨域两遍"

不二选一,把参照来源作为方法的显式自由度:
- **同一 selector 权重不变**,只换 witness 参照:训练域内用 last-action
  (单遍,零额外开销);零样本新平台用 proposal(两遍,RTF 12-20%);
- 全部现有数据直接支撑:desktop 两口径等效 + mobile 只有 proposal 存活,
  正好是"参照质量 × 平台距离"的二维故事,四点阶梯是机制证据;
- novelty 无损:零样本迁移声明仍成立(靠两遍),效率声明也成立(域内单遍);
- 无需任何新训练;剩余实验仅 OSWorld 满员后的配对与分层(今天内)。

**paper 主表形态**:mobile 主行 = 两遍(+6.41pp✓);OSWorld 主行 = 单遍
(vs recent 配对);消融 = 参照阶梯 + RTF 成本表 + mobile 单遍失效行
(作为"为什么跨域要两遍"的证据而不是败笔)。

## 4. 我的建议

**选 C**。理由:它把 A 和 B 的缺点互相抵消(A 的开销只在需要它的地方付,
B 的失效被显式解释并转化为机制证据),不新增实验风险,且叙事上"我们刻画了
witness 参照的可迁移性边界"比"我们选了某一个部署"更像贡献。若同事坚持
单一部署形态,则退 A(证据最硬),用 RTF 表防守成本质疑。

## 5. Provenance

- Mobile:`data/results/mobileworld_hgkv_selected_b4/`(per-task 五套 + 两遍
  开销审计 `two_pass_overhead_audit.json`);
- OSWorld:hyper01 `verified-sel1p-b4/`、hyper00 `verified-{recent-b4,b0}/`
  (满员后并集入库);
- 离线:`data/results/desktop_did_policy_v4/filltob_*.json`。
