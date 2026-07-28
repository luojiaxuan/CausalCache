# Rebuttal memo:OSWorld 上的 memory-critical 分层尝试(不入正文)

正文只报告 OSWorld 的总体边界(高保真历史 vs B0 有效;同预算不同 allocation
无显著差异,availability 而非 allocation 证据)。以下三个分层尝试均无显著
结果,保留于此备 rebuttal。

## 1. 域代理(multi_apps vs 其余)

LA−HGKV-recent:multi_apps(93)+2.15 [−3.23,+7.53];其余(268)
+0.75 [−3.73,+5.22];interaction +1.40 [−5.46,+8.32] p=0.72。
域归属 ≠ 记忆依赖(multi-app 可为纯顺序流程)。

## 2. 冻结双臂仪器(frozen_recent=1 且 b0=0)

该层(n=56)基线成功率 ~80% —— 仪器识别的是"**近期**记忆敏感"
(recent 已足够),不是"证据落在 Recent-B 之外"。
LA−recent 层内 −1.79 [−14.29,+10.71];interaction −3.43 [−15.60,+8.93]。
构念错位:没有任何臂组合能识别"旧证据不可恢复"。

## 3. 指令语义扫描(跨引用关键词)

361 条指令命中 31 条,人检大多假阳性("original message"、"come back
later" 等)。少数接近者(跨文档数据转移)也不满足严格定义,因为——

## 核心解释(rebuttal 用语)

OSWorld 任务指令自足(goal 完整给定)且证据可重访(文件持久、窗口可
重开),因此所需信息很少**不可恢复地**落在 recent 窗口之外;这与该平台
上 allocation 各臂无差异**一致**(consistent with re-observability;
该假说非事前冻结,不作 confirmatory 表述)。构造有效的 memory-critical
评测需要在基准设计阶段放置"证据在早前且不可重访"的任务,MobileWorld
的 memory_candidate/single_app_control split(62/55)正是,并给出
同预算 +8.6pp [+1.6,+16.1] 与 interaction +10.4pp p=0.022。

若审稿人要求 OSWorld 上的构造性子集:可人工审定 31 候选 + multi_apps
清单,预计合格 n≈10–20(欠功效),属 camera-ready/未来工作量级。
