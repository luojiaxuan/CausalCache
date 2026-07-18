# set-conditioned v3 首次 attempt

本目录记录 2026-07-18 的首次 v3 CPU-only attempt。formal-58 训练、五 checkpoint、fresh feature-only
prediction 和 label-blind seal 均完成；在持久化 label claim 并首次 decode fresh-16 label 后，轻量 historical
independent parser 因协议字符串绑定错误 fail closed。

错误只涉及 parser：实际 SHA-pinned artifact 使用
`causalcache_gate_v1_fresh16_evaluation_v1`，失败代码错误地要求
`causalcache_gate_v1_fresh16_primary_v1`。development report 未创建，因此本 attempt 没有 scientific result。
模型、预测、selector、阈值和 seal 在 label access 后均未变化，confirm/matched-NLL/closed-loop 保持 0。

Hyper00 文件只是 staging。后续只允许另立 parser-only repair，绑定本目录记录的 A/B、seal、claim 和 prediction
SHA，复用 sealed decisions，并如实报告 parent decode 1 次加 repair deterministic replay 1 次；不得把 repair
包装成 pristine one-pass evaluation。
