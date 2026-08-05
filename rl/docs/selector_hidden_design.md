# Selector v2:policy hidden-state 打分头(废除 28 维手写特征)

## 为什么换(用户 2026-08-05 裁定,数据早已支持)

cheap 塔的 28 维特征是链条上最后一块启发式:年龄/新近排名是位置规则,
witness 家族已实测无用(p_cheap 三项全最差),内容盲特征学不出"这帧对现在
有没有用"(方案3:embedding 相似度与随机不可分辨)。cheap 塔存在的唯一理由
"部署零成本"在 RL pivot 后已死——server 每步本来就在跑 GUI-Owl。

## 架构:外围视觉(索引遍)+ 注视(动作遍)

```
索引遍(新,冻结策略,一次前向):
  输入 = 指令 + 每个历史事件[动作行 + 缩略图 ~144 token] + 当前屏缩略图
  输出 = 各历史图 token 段的 mean-pool hidden state h_i(H 维,模型自己的表征)
  打分 = 线性头 w·h_i(可学参数 ~H+1 个)→ Plackett-Luce 采样 B 帧(τ 探索)
动作遍(不变):
  选中 B 帧全分辨率(2560 token/图)+ 当前屏 → 冻结策略 + HGKV LoRA → 动作
```

- 30 步历史 × 144 token ≈ 4.3K 视觉 token,50 步 ≈ 7.2K,上下文无压力;
- "候选动作 + 历史事件摘要"= 每步动作行(官方渲染),已在索引 prompt 里;
- 年龄/新近不再是手写特征——它们以位置编码的形式存在于模型自己的表征里,
  要不要用由 RL 决定;
- **部署成本诚实披露**:每步多一次 ~5-8K token 的 prefill(约 1-2s),
  低于旧 CausalCache-P 的全分辨率 pass-2。

## v1 边界(记录,不隐瞒)

1. 打分头只有线性层,索引遍**不挂 LoRA**——头学不动再上索引遍 LoRA(v2);
2. PL 采样对固定分数做无放回顺序抽取,**没有集合上下文重打分**
   (cheap 塔的 set-context 特征取消;索引遍本身看全集,上下文在表征里);
3. 审计存**池化后的 hidden 向量**(fp16 base64,~240KB/步·30 候选),
   trainer 只训头时从向量复算 log π,零环境重建——与 cheap 模式同一机制,
   "特征"从手写 28 维换成模型自己的 H 维。

## 管线变更面

| 件 | 变更 |
|---|---|
| serve | `--selector-mode hidden` + `--selector-head` + `--index-visual-tokens`;cheap 模式原样保留 |
| 审计 | 行内加 `mode:"hidden"` 与 `hidden_vectors{event: fp16_b64}`;rounds 只存 {candidates, chosen, logprob} |
| collector | 透传 hidden_vectors |
| trainer | bundle kind 分支(hidden_head = Linear(H,1));log π 复算从 hidden_vectors 解码 |
| bootstrap | `make_hidden_head.py` 随机初始化头(std 0.02),H 从模型 config 读并断言一致 |
