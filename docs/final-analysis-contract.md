# 终版分析契约(冻结于 2026-07-28,先于补充实验提交)

依据与同事的路线裁决(A 主文、B 消融、C 落地)冻结以下口径。本文件提交
时间早于下述任何新增 run 的启动时间,用 git 时间戳自证非 post-hoc。

## 1. 方法命名与冻结组件

- **CausalCache-P(主方法)**:HGKV + selector,witness 参照 = pass-1 拟议
  动作(proposal),两遍推理,默认/迁移模式;
- **CausalCache-LA(效率变体)**:同一权重,参照 = 上一步已执行动作,单遍;
- 全部主表行冻结同一套组件:HGKV checkpoint `lora-step300.pt`
  (sha256 5720…67a9);selector **two_tower** gold-trained
  `marginal_scorer_twotower.pt`(sha256 前缀 c72974ab611c);B=4,beam=3。
  P/LA 仅改变参照来源。concat、native-lastact、intent、no-witness 全部只进
  ablation。

## 2. 主对照(primary contrast)与基线层级

- **Primary**:CausalCache-P − (HGKV + Recent-B4),同平台逐任务配对;
- Secondary:vs Frozen B0、vs Frozen+Recent(mobile 已有官方轮);
- 条件加入:Frozen + Selected-P(adapter × selector 2×2 分解)。

## 3. 轮数与检验(mobile)

- (2026-07-28 修订,早于 recent r2/r3 完赛:)各主臂**等轮数 = 2**:
  primary = P(r1,r2) vs HGKV+Recent-B4(r1,r2)双侧配对;P r3 取消;
  recent r3(与 r2 并行已启动)完赛后仅作 robustness 附注,不进主检验;
  Frozen B0 三轮作 secondary(基线侧超采样,披露);
- 检验:任务级配对均值差,**双侧** cluster bootstrap 10k(按任务重采样),
  报告点估计、CI95、双侧 p;不把四舍五入 CI 界写成强结论;
- 分层报告(描述性):memory_candidate(62)/single_app_control(55)。

## 4. OSWorld 补格

- 运行 CausalCache-P 全量 361(与 LA/recent/B0 同名册、同 policy、同
  selector);形成 closed-loop 2×2(P/LA × OSWorld/Mobile);
- Primary contrast 同上:P − (HGKV+Recent-B4),双侧配对。

## 5. 成本仪表(逐任务,随 P 运行落盘)

每任务记录并在 paper 报告:额外 policy calls 数、pass1/pass2/selector
model-seconds 合计、端到端墙钟、P50/P90;成功任务与全部任务分别统计;
两个口径并列:policy-compute overhead(≈+66%)与 end-to-end RTF。

## 6. 表述降级(与数据一致)

- 参照阶梯改为:action-conditioned(gold/proposal/last-action)整体优于
  instruction-only/none;gold 与 last-action 相对顺序依平台与协议而变;
- gold(离线 teacher-forced oracle 上界)与 proposal(在线)分开陈述;
- mobile P vs B0 在等轮数分析落地前表述为 borderline paired evidence;
- LA 的 mobile 失效作为主动披露的负结果 + "参照是迁移瓶颈之一(与该
  解释一致)",不声称已刻画迁移边界。
