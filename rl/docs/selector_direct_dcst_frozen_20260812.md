# Direct DCST——模型设计冻结版(2026-08-12,外部方案 v2,用户裁定)

> Direct Counterfactual Subset Transformer。核心一句话:**给定当前状态与
> 候选历史帧集合 S,直接预测"用 S 替换 recent-B"对冻结策略准确率的净
> 收益。** subset-native,B 是实验配置不是网络结构;主实验 B=2(≤C(30,2)
> =435 全枚举,无搜索近似)。相对上一版(DCET)的两个改动:①pass-1
> 不进主模型(降为增强消融)——部署每步只有一次策略前向;②B 不写死。

## 输入输出
s = (instruction I, 动作历史文本 H, 当前屏 A, 候选帧集 F(2≤N≤30), recent-B R)。
帧输入 = 冻结缓存 X_i [~2584,4096];当前屏用同一冻结特征口径。
学 G_θ(S;s) = 相对 recent-B 的预测收益;S* = argmax G;G(S*)>τ 才 SWITCH。

## 架构(选帧器全内部无任何维度写死为 2)
1. **High-Fidelity Frame Resampler**:LN→W_in(4096→512)+E_spatial(真实
   二维 token 坐标);K=128 latent/帧(96 spatial-anchor + 32 global),
   2 blocks,8 heads,FFN 2048,dropout 0.1,BF16。禁:缩略图、mean pool、
   单 CLS、单帧向量打分。K=64/32 留作后续轻量化消融。
2. **当前屏**:同一 resampler,128 latent;selector 与最终策略前向之间
   复用编码(工程允许时)。无 autoregressive draft。
3. **Context Encoder**:GUI-Owl 冻结 text token embedding → 512 投影 →
   2 blocks;按 step 压缩为 [c_task, c_1..c_{t-1}],保留 8-32 个 token。
4. **Frame metadata**:帧年龄 Δt、绝对 step、对齐动作表示、是否∈recent-B。
   允许真实时间;禁用候选数组位置(置换等变,打乱候选预测不变)。
5. **Candidate Set Encoder**:u_i=AttnPool(L_i);z_i=u_i+E_age+E_action;
   3 blocks Set Transformer(条件于 C)→ H_i。不输出帧分,禁 s_i+s_j。
6. **Counterfactual roles**:S∪R 按 ADD(S\R)/RETAIN(S∩R)/REMOVE(R\S)
   加 role embedding;当前屏 E_CURRENT。学的是记忆干预 R→S。
7. **Counterfactual Subset Transformer**:M_S=[L_ADD,L_RETAIN,L_REMOVE,
   L_current,C,H_S,H_R];Q=8 learned subset queries;cross→self→cross
   两轮;z_S=AttnPool。子集内全部 fine latent 同场 attention →
   G({i,j}) 与 G({i})+G({j}) 无关(涌现组合可表达)。
8. **输出分解**:
   - Recent-Failure Head:q_s=P(b=0|R,current,C),每态一次、全候选共享;
   - Rescue:r_S=P(y_S=1|b=0,s,S);Harm:h_S=P(y_S=0|b=1,s,S);
   - **G(S)=q_s·r_S−λ(1−q_s)·h_S,primary λ=1;G(KEEP)=0;G>τ 才动**
     (τ 只在内层 OOF 校准,primary 只校 τ 一个自由度)。

## 损失(batch 单位=整个 state)
- L_base = BCE(q_s, 1−b);
- L_conditional:b=0 → BCE(r_S,y_S) 态内平均;b=1 → BCE(h_S,1−y_S) 态内
  平均;**先态内平均再跨态平均**(78 边的态不得压过 8 边的态…组合数
  加权禁止);
- L_deploy(结构化对齐部署 argmax):
  b=0:g+=SmoothMax{G:y_S=1}, g−=SmoothMax{G:y_S=0};
       L=softplus(m−g+)+softplus(m+g−−g+);
  b=1:L=softplus(m+SmoothMax{G:y_S=0})(正确替代不强压,τ 会 KEEP);
- 总:L = L_base + α·L_cond + β·L_deploy,α=β=1 起步;不稳则先训概率头
  2-3 epoch 再加 L_deploy;SmoothMax 温度退火趋 hard。

## 训练协议
外层 5 折沿用固定协议(排序→seed 20260809 洗牌→20% 切片);内层验证做
早停/τ/极少量超参;测试折只跑一次。增广:候选置换(等变自检)、
candidate dropout(非 subset/非 recent);禁几何类增广。模型选择:
内层 (selector−recent) paired diff 为主,W−L 次之;禁按 pair AUC/BCE 选。
AdamW,骨干 0.5-1e-4 / 头 2-3e-4,wd 0.03-0.1,BF16,clip 1.0。

## 评测(5 折 OOF n=1263)
主表:recent acc / selector acc / paired diff / 态 bootstrap CI / McNemar /
W / L / **W/L>1** / move rate / switched hit rate / MultiApp / 单应用。
历史失败共同特征 W/L≈1/2,重点盯 W/L 翻转。

## 消融(首轮只做四个)
Full / −joint subset interaction(非加性作用)/ −counterfactual roles /
absolute correctness head(gain+KEEP 设计必要性);资源允许再加
+pass-1 draft(行为信号增量;**不属于 Direct DCST 定义**)与 K=64/128。

## 硬约束(写进实现 spec)
1. 禁任何加性帧分 G(S)≠Σg(F_i);2. 最终决策前禁把帧压成单标量;
3. Subset Encoder 必须一次看到 S 的全部 fine latent;4. recent-B 显式进
scorer 且带 ADD/RETAIN/REMOVE role;5. KEEP=显式零收益,不得强制每态
换子集;6. 接口禁固定 pair-position 参数(score_subset(candidate_subset,
baseline_subset, current, context),不存在 score_pair)。B 泛化=同结构
不同 checkpoint,不要求 joint multi-B/zero-shot。

## 执行侧核验与偏差记录(实现方,2026-08-12)
- 训练监督:labels_all 的帧对表直接可用 ✓(无需任何新策略前向);
- **部署评测例外**:部分态枚举当年剪枝,argmax 选到未标注子集时用既有
  补测管线(rl_score_chosen_subsets,冻结策略真跑)补齐后再判 ——
  设计不变,评测流程保留补测步;
- E_spatial 的二维坐标:缓存未存网格形状,由 token 数反推(1920×1080→
  38×68=2584 精确整除;非标屏取最接近 16:9 的因子分解),记为实现近似;
- E0(pass-1 门探针,AUC 0.696)保留为独立诊断结论,不进主模型。
