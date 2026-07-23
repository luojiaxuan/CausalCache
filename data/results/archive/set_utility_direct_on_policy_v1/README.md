# Direct on-policy coverage v1

状态：`COMPLETED_LABELS_AND_MERGED_TRAINING_INPUT`。

冻结的 direct-v3 checkpoint、recent 与 confidence-gated hybrid 已在全部 10,658 个 train states 上完成
rollout。Hyper00/Hyper01 共 11 workers，10,658 个 state id 全局唯一且与 assignment 逐项一致；无缺失、
无重复、无 worker failure。首次 launcher 因 `PYTHONPATH` 缺失在 import 前失败，没有生成 selection；修复后
使用全新 output root 重启，失败目录不进入本结果。

missing-only schedule 只针对此前没有任何 complete conditional group 的 5,108 states。它补齐三条部署路径
在 `|S|=0,1,2,3` 实际访问 base 的全部 `S∪{j}`，因此包含 359,189 个缺失 coalition，而不只是 24,787
个 empty-to-singleton 下界。分布为 medium/long/very-long=`3,578/1,499/31`；完成后 complete groups 将从
62,332 增至 120,172。该数字后来确认是 schedule-time projection：它把已有 complete group 的 states 上
并未实际调度的 desired coalitions 也计入了结果。merged bytes 的真实 full-train census 为 96,759 groups；
其中 9,287 optimizer states 有 84,441 groups，训练合同以后者为精确 optimizer inventory。

正式 label rollout 已在 Hyper00/Hyper01 闭合：5,108/5,108 states、33,175/33,175 microbatches、
359,189 个实际 teacher distance rows，加上 5,108 个不触发 forward 的 `D(C)=0` anchors；0 skip、
0 non-finite、0 duplicate rows、0 bad logs。terminal content SHA256 为
`689b3ad3cdf0fafdd3e3a40979aa86109eae40f467c6ade1551e7bf530c67a0e`。

这些 labels 已与 frozen base input 合并。新 training input content SHA256 为
`6c9243a2a2846180f717ea59e3692a3005ae153b8d206f7ad1ec4a8bf1a1bfad`；5,108 个重复 full anchors 的
最大差异为 0。formal optimizer inventory 固定为 900 trajectories、9,287 states、84,441 个
candidate-complete groups，另有 256 个 trajectory-disjoint checkpoint-selection states。两份 fresh-init
训练配置已绑定同一 input 与 inventory；每个 epoch 后必须完成真实 held-out B1--B4 recovery，才可继续训练。

初版 missing-only schedule 未列出 label runner 会默认注入的 full-history `D(C)=0` anchor，导致 workers 在
首个 state 的 forward 全部完成后无法通过 terminal set-equality check，0 terminal。v2 schedule 已为每个 state
显式增加一个 zero-cost、不会进入 teacher forward 的 full anchor；359,189 个待计算 labels 没有变化，旧 schedule
与失败 output 均不进入训练 artifact。

机器可读摘要见 [`summary.json`](summary.json)。大 artifact 当前位于摘要记录的 persistent paths，状态为
`PENDING_HF_UPLOAD`；后续与 checkpoints 一并发布 immutable Hugging Face revision。
