# CausalCache History-Gated Mainline V1(冻结契约,2026-07-23)

完整设计见本文;要点契约如下,执行清单见第 15 节。main 保底路线不受本分支影响。

## 核心问题
在完全冻结 GUI-Owl 原始参数的条件下,只新增一个解释历史视觉证据的接口,能否从
benchmark-external 轨迹(GUI-Odyssey 修正版)零样本迁移到 AndroidWorld 与 OSWorld。

## 行为契约
- 无高保真历史(B0):严格等于 frozen GUI-Owl(**bitwise parity 或预注册数值容差;
  架构不变量,训练前必须通过,失败不得开训**);
- 错误/无关历史:尽量回退 frozen 行为;
- 正确且有用历史:允许朝成功动作方向改变分布。

## V1 架构:History-Gated KV Adapter
- 冻结:vision encoder / projector / 全部原 LM 参数 / tokenizer / tool schema / parser / executor;
- 位置:LM 最后 8 层(28-35),不 sweep;
- 模块:仅 k_proj/v_proj,仅作用于 restored history image tokens(mask 门控 residual):
  K_l = W_K h + M_hist·ΔW_K h;V 同理;
- 配置:rank 8, alpha 16, dropout 0;
- B0:history mask 全 false 时 hook 直接返回原输出,不执行 LoRA residual。

## Token-role contract
新增 history_adapter_context.py(ContextVar,防并发串 mask)、history_token_roles.py、
history_gated_lora.py。图像顺序固定:恢复历史图 1..K,当前观测图,K=0 即 B0。
mask 由 mm_token_type_ids + image_grid_thw + restored_event_step_ids 构造;禁止固定
token ID/经验 offset/硬编码 visual 数猜角色。fail-closed:图块数 ≠ K+1、mask 触及当前图/
目标 token、image-grid 不一致、恢复事件数不一致。

## 训练目标(ℓc/ℓ0/ℓs/ℓi;π0=frozen,πθ=frozen+adapter)
- L_gain = [m_g − (ℓc − ℓ0)]+(主目标;直接防止 correct<b0 旧病);
- L_rank = [m_r − (ℓc − ℓs)]+ + [m_r − (ℓc − ℓi)]+;
- L_ignore = SmoothL1(ℓs, ℓ0) + SmoothL1(ℓi, ℓ0)(V1.1 再考虑 action 段 top-k KL);
- L_norm = ‖ΔK‖² + ‖ΔV‖²;
- L_CE 权重 0(或 0.05),不得回到强 CE 主导;
- 初始:gain/rank/ignore 权重 1.0,norm 1e-4,margins 0.01。

## 数据协议
- 修正版 GUI-Odyssey;正式 manifest 必报:轨迹/完整轨迹/决策态/终止态计数、逐动作计数
  (含 Home/Back)、全部 drop 原因计数;terminal 每成功轨迹一次,不过采样;
- 变体 correct/b0/shuffled/irrelevant(可选 missing-critical),共享同一成功动作 target;
- 轨迹 ID 三切分 train/tune/heldout,任何 state/截图/donor 不跨 split;
- checkpoint 只按 Odyssey tune/heldout 选;
- AW/OSWorld 各设 development roster(仅 canary/调试)与 sealed roster(结构/loss/选择协议
  冻结前生成,正式结果唯一来源)。

## Adapter 正式 gate(全 CI 下界 >0)
correct−frozen-b0、correct−shuffled、correct−irrelevant 全过 + wrong-history drift 有界 +
B0 parity 精确。未过不得进 selector label 生成。

## 零样本 policy 评测矩阵
{Frozen, Full-layer LoRA, History-gated} × {B0, Recent-B, Full history} 于 AW 与 OSWorld;
不许按 benchmark 结果改 adapter/选 checkpoint;canary 只抓工程故障。

## V2(仅当 V1 出现 correct>shuffled/irrelevant 但 correct 不稳超 B0)
增 gated Q/O readout(仅 history_present 时启用)与可选 role embedding;B0 仍严格关闭。

## Selector(adapter 过跨平台 gate 后)
U_act(S) = log p_gate(a*|S) − log p_0(a*|B0);第一版 per-event singleton gain +
positive-gain threshold + at-most-B + STOP;特征仅平台通用;GUI-Odyssey-only 训练与选择,
零样本部署;Odyssey heldout 上须稳定胜 Recent/Similarity/Random(recovery、oracle regret、
positive-gain precision)后才进闭环。

## 升级/回退(Outcome A-E)
A:双平台正向 → 升级主方法,full-layer 降为 baseline;B:adapter 双平台成、selector 仅 AW →
主方法保 policy 部分;C:仅 AW 成 → 收缩 mobile transfer;D:离线成闭环不升 → 机制分析 +
进 V2;V2 仍败 → E:封存,main 路线回归,history-gate 仅作失败分析。

## Abstract 关系
探索期不改;成功后 "adapt a GUI policy" → "attach a policy-preserving history interface
to a frozen GUI policy";跨平台闭环出结果前不写 zero-shot across 句。

## 第 15 节执行清单(顺序,不并行扩方法复杂度)
1 建分支 ✓;2 冻结本文档 ✓;3 Adapter API 重构(保 full-layer 兼容);4 token-role mask;
5 top-8 history-KV gated LoRA;6 B0 bitwise parity;7 prefill/decode/并发测试;8 小样本 smoke;
9 Odyssey 正式训练;10 仅按 Odyssey heldout 冻结 checkpoint;11 双 benchmark dev canary;
12 sealed 零样本 policy 评测;13 过门后重标;14 Odyssey-only selector;15 sealed policy+selector
评测;16 按决策树定主线。
第一阶段唯一问题:B0 完全不变前提下,正确历史能否在 Odyssey heldout 显著优于 B0/shuffled/
irrelevant。此问未过,不做复杂 selector、不引 desktop 训练数据、不按 benchmark 改 adapter。
