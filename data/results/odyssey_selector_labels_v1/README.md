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
hyper00 /data02/jaxan/runs/ody-selector-labels/uact_labels.jsonl(每行:pair_group、b0_logprob、
candidates{K: single_logprob, u_act});PENDING_HF_UPLOAD。打分 checkpoint odyv2-step75
(sha256 75670fa5...);frozen b0 参考同 GUI-Owl 冻结基座。
