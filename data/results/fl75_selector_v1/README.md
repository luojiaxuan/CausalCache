# Odyssey selector U_act 标签 v1

在 history-aware policy(odyv2-s75,坐标修复版)上生成 success-anchored 选择器标签。

## 定义
U_act(j) = log p_s75(a* | 恢复候选事件 j 的单图) − log p_frozen(a* | b0 纯摘要)
a* = GUI-Odyssey 人类演示的成功动作(teacher-forced)。

## 规模(分母全对账)
- 10,680 个决策点(Odyssey train,shared-early 后的记忆步);
- 75,628 个 (决策, 候选) 对,每决策最近 ≤8 个候选;
- b0 参考 10,680 个,0 缺失。

## 分布(信号强,支持训练)
- 82.8% 候选 U_act > 0(恢复历史图普遍提升成功动作概率),均值 +0.159 nats/token;
- 候选间 std 0.164(≈均值量级)——**贡献有区分度,selector 有实质选择空间**;
- 85.3% 决策点至少一个正贡献候选。

## 对比
冻结 policy 在 AndroidWorld 上内容盲(U_act≈0);Odyssey-trained s75 在自身域 U_act 强正且
有区分度——history-aware 适配让选择器标签"活化",验证了整条 policy-then-selector 路线的前提。

## 数据
hyper00 /data02/jaxan/runs/fl75-selector-labels/uact_labels.jsonl(2026-07-24 改名)(每行:pair_group、b0_logprob、
candidates{K: single_logprob, u_act});PENDING_HF_UPLOAD。打分 checkpoint odyv2-step75
(sha256 75670fa5...);frozen b0 参考同 GUI-Owl 冻结基座。

## 线路归属与命名(2026-07-24,用户裁定)

**本目录全部产物属 full-layer s75 policy 的 selector 臂(消融矩阵行),非 paper 主线。**
主线 selector 挂 history-gate adapter(分支 exp/history-gated-mainline-v1 契约第 11 节,
标签须由过门的 gated policy 生成)。命名规约:主机 run 目录与文档前缀 `fl75-` = 本线,
`hgkv-` = history-gate 主线;hyper00 目录已改名 /data02/jaxan/runs/fl75-selector-{labels,emb,v1*}。

## V1 多模态 selector 结果与机制发现(fl75 线)

- V1(s75 query 三段池化 + vision post/delta + 三头 + Huber/rank/sign):heldout 组内
  Spearman +0.023,top-1 0.159,B4 captured 0.401——**不敌 recent**(+0.059/0.286/0.609);
  visual-sim、random 亦与 recent 无实质差异。
- **方差分解定案**:U_act 总方差的 **99.2% 在状态间,组内仅 0.8%**(组内极差中位数
  0.012 nats,单前向噪声底附近)。"选哪张图"在本 regime(Odyssey/s75/≤8 近邻候选)
  无可学信号;"该状态恢复历史有没有用"才是信号所在。
- **状态级门控可行性**:线性 ridge on s75 b0-prompt hidden(三段池化)预测状态平均增益,
  heldout Spearman 0.456 / R² 0.214(alpha=1e4)。
- fl75 线 selector 落地形态:**状态门控 + recency 排序**(gate 超阈值才恢复 recent-B,
  否则 B0)。待做:MLP gate + 阈值标定 + 闭环集成 + test-25 终局矩阵。
- 产物:hyper00 /data02/jaxan/runs/fl75-selector-v1-main/(V1 checkpoint + 报告 + 基线),
  fl75-selector-emb(query 10,680 / events 15,680),PENDING_HF_UPLOAD。
