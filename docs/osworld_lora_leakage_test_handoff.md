# OSWorld LoRA 渗漏测试——交接文档(margin-SFT 线 → OSWorld runner 线)

给 OSWorld runner 会话:请用你们已跑通的 runner 执行本测试,结果写回
`docs/osworld_lora_leakage_v1.md` + `data/results/osworld_lora_leakage_v1/` 并推 main。

## margin-SFT 线现状(你们可能未同步的部分)

- **AW-trained v3-e1**(rank-16 全 LM 层 q/k/v/o LoRA,α32):heldout 内容门禁 CI 过线;配对闭环
  化验尺(retry 协议)**adapted-B0 vs frozen-B0 = +8.9pt CI[+0.022,+0.156]**(首个 CI 认证闭环提升);
  记忆剂量边际(B8−B0)未显著。
- **Odyssey-trained s75**(GUI-Odyssey 900+ 人类轨迹,75 优化步):AW 零样本内容门禁 CI 过线
  (c−shuf +0.0070[+0.0052,+0.0091]);但首次零样本闭环 0/90——根因 = 训练目标 **0 个 terminate**
  (state 母表只含中段决策点),已修:终止态合成 3,233 样本 + 从 s75 混合续训(hyper00
  `/data02/jaxan/runs/causalcache-odyssey-margin-v1b/lora-step{20,40,60}.pt`,今晚落盘)。
- 教训直接催生本测试:**CE 会重塑动作分布**;mobile_use 语法内 terminate 被饿死了,desktop 的
  computer_use 语法是否被 mobile 训练渗漏,未测。

## 测试规格

**问题**:mobile-语法 LoRA 是否渗漏进 computer_use 生成(desktop prompt 下语法/动作分布劣化)。

**材料**:
- checkpoints(hyper00,61MB/个;跨机传输**务必 ssh 直传 + 双端 sha256**,不要经带 CUDA banner
  的镜像 cat,会污染文件——我们踩过):
  - 主角:`/data02/jaxan/runs/causalcache-odyssey-margin-v1/lora-step75.pt`
  - 终止修复版:`/data02/jaxan/runs/causalcache-odyssey-margin-v1b/lora-step60.pt`(落盘后)
  - AW 版对照:hyper01 `/data02/jaxan/runs/causalcache-margin-sft-v3-eval/lora-epoch1.pt`
- 加载:`scripts.train_success_sft_lora` 的 `inject_lora`(rank 16, alpha 32, q/k/v/o,排除 visual)
  + `load_lora_state_dict`;α 缩放 = 加载时把 alpha 传 16(半强度)。参考
  `run_androidworld_memory_ceiling_worker.py` 的 --lora-checkpoint 段。

**流程**:≥30 个 desktop prompt(跨域,含多图 recent-B4 恢复形态),贪心生成,条件:
frozen / +s75(α32) / +s75(α16) / +v1b-s60(α32)。

**指标**:
1. 语法合法率(runner 的 normalized parser 通过率);
2. 动作类型分布 vs frozen(KL 或计数表);
3. `click` vs `left_click` 混写率(冻结模型已有此病,看是否加剧);
4. **硬渗漏标志**:输出中出现 mobile_use 词汇(swipe / long_press / system_button / open)即判硬渗漏。

**判定规则(预注册)**:
- 合法率降幅 <2pt 且零硬渗漏 → PASS:OSWorld 可用 LoRA 版,双 benchmark 主结果成立;
- 降幅 2-10pt → α16 复测,过则用 α16;
- 降幅 >10pt 或任何硬渗漏 → FAIL:OSWorld 只报冻结 policy 诊断(内容盲/干扰税),LoRA 结果限 mobile。

## 我们这边的并行事项(供对齐)

终止修复续训完成后:10 局 B0 迷你冒烟(验终止行为)→ 重发 AW 零样本化验尺 90 局。paper 主表
在 AW 侧收敛;OSWorld 侧按你们测试结果定叙述档位。progress.md 有全量时间线。

## 更新(2026-07-23):测试主角改为 AW-trained v3-e1

- Odyssey 线闭环两连败(0/90×2,见 data/results/odyssey_zeroshot_closed_loop_v1/),其 checkpoint
  不再是 OSWorld 测试候选。
- **测试主角:AW-trained v3-e1**(闭环 +8.9pt CI 认证的那个):
  - hyper00 `/data02/jaxan/runs/causalcache-margin-sft-v3/lora-epoch1.pt`
  - hyper01 `/data02/jaxan/runs/causalcache-margin-sft-v3-eval/lora-epoch1.pt`(sha256 前缀 7122d897)
- 建议测试条件:frozen / v3-e1(α32) / v3-e1(α16) 三条件;指标与判定规则不变(语法合法率 /
  动作分布 / click 混写 / mobile 词汇硬渗漏)。若语法过关,可加跑少量 OSWorld 闭环任务看真实表现。
