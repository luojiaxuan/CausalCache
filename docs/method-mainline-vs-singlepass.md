# Context 选择方案对比:两遍主线 vs 单遍支线(2026-07-28,与同事讨论用)

## 0. 共享底座(两方案完全一致的部分)

- **Policy**:GUI-Owl-1.5-8B + HGKV(history-gated KV LoRA,desktop v4 s300),
  官方多轮渲染;非连续历史帧用 gap-fold 在真实时间位恢复(单测锁定:连续尾
  退化 byte-exact 等于官方 builder)。
- **Selector**:统一集合条件边际打分器 Δ̂(j|S)(cheap 20 维 + set-context 8 维,
  two_tower/concat),exact-B beam-3 填满、recent 兜底。
- **唯一分歧点**:witness 特征族(4 cheap + 2 set)的"目标动作"从哪来。
  witness = "候选帧的后续动作 ≈ 目标动作"(全动作等价:类型+按钮+坐标容差 25)。

## 1. 方案 A:两遍 propose-then-select(当前 paper 主线)

目标动作 = **pass-1 的拟议动作**(训练时用 gold teacher target)。部署:先按
Recent-B 出一遍动作 → selector 用它算 witness → 若换帧则 gap-fold 重组 prompt
再出一遍;不换帧直接用 pass-1 结果。

| 口径 | B=2 | B=4 |
|---|---|---|
| two_tower | +0.0250(显著) | **+0.0108 [+0.001,+0.021]** |
| concat(desktop 主臂) | **+0.0297**(显著) | **+0.0169**(显著) |

- 闭环(MobileWorld,two_tower):selected-B4 **39/117=33.3%**,vs 冻结 B0
  +5.13pp(CI[−2.6,+12.8] p=0.29,单轮欠功效,加轮在跑)。
- 开销(线上审计已积累):selector 本体 ~6ms;pass-2 仅在换帧时触发,prompt
  同预算(B 张图)→ 最坏 2× 生成,期望远低于(触发率分布 r2 跑完给出)。
- 优点:witness 信号最强(直接对齐"将要做的动作");机制故事完整。
- 缺点:两遍推理需要向审稿人辩护;部署路径复杂(propose→select→re-decide)。

## 2. 方案 B:单遍 last-action witness(2026-07-28 新结果)

目标动作 = **上一步已执行动作**(选记忆前已知,零额外前向)。机制解释变为
"**近因重现**":当前动作模式上次出现的位置就是值得恢复的旧帧(GUI 操作的
循环性:同一控件/区域反复交互)。

| 口径(打分器仍为 gold-witness 训练版,直接换特征口径) | B=2 | B=4 |
|---|---|---|
| last-action 单遍 | **+0.0247 [+0.0133,+0.0363]** | **+0.0152 [+0.0058,+0.0250]** |

- **两个预算全显著,B=4 点估计甚至高于两遍口径**(+0.0152 vs +0.0108
  two_tower;差异在噪声内,但至少不弱)。
- k 分布健康(B=4:k=0 占 29%,k=4 占 19%,无 nowitness 的过度替换病理)。
- **原生重训版在跑**(训练特征=部署特征,消除口径错配,预计更强)。
- 开销:selector ~6ms,**无 pass-2,单遍出动作**——部署故事最干净。
- 风险/注意:
  1. 这是第三个单遍变体(nowitness、intent 之后),存在 post-hoc 选择嫌疑;
     但 CI 下界 +0.0058 距零有余量,且原生重训 + 闭环验证可作独立确认;
  2. 闭环未验证(现有 39/117 是两遍臂);mobile 侧移植极轻(server 里把
     witness 伪目标从 proposal 换成 last executed action,去掉二遍分支);
  3. 机制叙事从"意图对齐"变为"近因重现",与 DiD 训练目标(重现帧高保真
    恢复有因果收益)其实更自洽。

## 3. 参照系:另两个单遍变体(下界)

| 口径 | B=2 | B=4 | B=4 病理 |
|---|---|---|---|
| witness 清零(nowitness) | +0.0161(显著) | +0.0086(跨零) | k=4 暴涨 34% |
| + intent v2(指令条件 8 维) | +0.0149(勉强显著) | +0.0046(跨零) | k 正常但无增益 |

三个单遍变体 + 两遍构成一条干净的机制曲线:**目标信号质量
gold/proposal > last-action > intent > none 单调对应选择收益**——这本身就是
witness 机制的证据链,四点都可进消融表。

## 4. 决策建议

**建议以方案 B(单遍 last-action)为部署主线,方案 A 降为 upper-bound 消融**,
理由:
1. 效果不弱(离线 B=2/B=4 全显著,B=4 点估计 ≥ 两遍);
2. "零额外推理开销的即插即用记忆重排"是比"两遍推理换显著性"强得多的贡献
   叙事,reviewer 攻击面小;
3. 消融表反而更完整:两遍 gold-witness 是自然上界,intent/none 是下界,
   主线落在可部署的中间点。

**采纳 B 需要补的实验**(约 1–1.5 天):
- 原生重训确认(在跑,今晚出);
- MobileWorld 闭环单遍臂(server 改动小时级,campaign 复用 hyper01 舰队);
- (可选)OSWorld test_small 上同表(基建已half-way)。

若同事倾向保守(A 为主线),现有 paper 文本已经自洽(两遍正当性段落 +
intent 失败作为 dissociation 证据),只需把 last-action 加进消融表。

## 5. 数据与 provenance

- 离线判定文件:`data/results/desktop_did_policy_v4/filltob_intent_singlepass.json`、
  `filltob_lastaction_singlepass.json`(本次)、`filltob_{two_tower,concat,nowitness}.json`;
- 伪目标实现:`code/causalcache/selector_v4_features.py`
  (`CAUSALCACHE_WITNESS_PSEUDO_TARGET=last_action`,单测:norm999 坐标对齐);
- 闭环:`data/results/mobileworld_hgkv_selected_b4/README.md`;
- 两遍开销审计:arm 分支 8e01527(`SELECT_AUDIT` 逐请求日志,r2 积累中)。
