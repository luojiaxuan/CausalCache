# AndroidWorld memory ceiling v1(正式裁决)

状态:`COMPLETED_180_OF_180 / STORY_DEAD_CAPABILITY_BOUND`(预注册规则:B8−B0 模板级净胜 0 ≤ +1)。

15 个记忆密集型 dev 模板 × 3 实例 × 4 剂量臂,180/180 episodes,infra 仅 2。冻结协议:共享
summary-only 前 2 步、2,560 token/图、`76230e4`。

| 臂 | 成功 | 模板 macro | parse 死亡 | 循环局 |
|---|---:|---:|---:|---:|
| summary_B0 | 7/45 | 0.156 | 10 | 14 |
| recent_B2 | 4/45 | 0.089 | 19 | 12 |
| recent_B4 | 5/45 | 0.111 | 17 | 8 |
| recent_B8 | 7/45 | 0.156 | 13 | 6 |

`B8−B0`:模板级 2 胜 2 负 11 平,net=0,成功率差 `+0.000 [−0.089, +0.089]`。**在 GUI-Owl-1.5-8B
上,把高保真记忆开到上下文可行上限,closed-loop 成功率没有系统收益。**

## 诊断(三个互相抵消的真实效应)

1. **行为确实分叉**:43/45 cell 的动作序列在前 4 步内分歧("图被忽略"不成立;第 1–2 步分歧含环境
   渲染噪声)。
2. **过程收益单调存在**:动作死循环局随剂量下降 14→12→8→6——高保真历史抑制原地打转。
3. **两笔税吃掉收益**:(a) 多图 prompt 使严格 tool-call 语法的单步 parse 失败率约 3 倍
   (0.008 → 0.018–0.024),一击毙命协议下记忆臂 29–42% 的局死于 parse(B0 22%),且优先杀死长局;
   (b) 幸存失败以 step_budget_exhausted 为主——24–60 步任务需要 ~95%+ 每步执行可靠度,策略不具备,
   属能力地板,记忆不可修。

结合离线结论(oracle 恢复 headroom +0.375~0.47)构成 **dissociation finding**:行为保真可大幅恢复、
过程指标可测改善,但在 ~25% 能力水平的 backbone 上不转化为任务成功。

## 后续选项(按已识别机理)

- **协议修正 v2(便宜,1.5h)**:parse 失败允许单次重试(官方部署即如此),移除对记忆臂的不对称
  格式税后重测;v1 裁决按纪律保留;
- **能力提升弧线**:rejection-sampling SFT(LoRA)/ 更大 backbone,以本实验为 1.5 小时化验尺筛
  "B8−B0 是否张开";
- paper closed-loop 章按 dissociation finding 撰写。

## Artifact

- [`summary.json`](summary.json)(content SHA `da1a7e46...`);
- 逐 episode records(180 files,含逐决策动作/记忆选择/parse 结果/生成 token):**`PENDING_HF_UPLOAD`**
  (本会话权限策略禁止读 HF token,非 HF 故障)。已按上传字节暂存 hyper01 持久盘:
  - path:`/data02/jaxan/staging/androidworld-memory-ceiling-v1-76230e4/`
  - files:`episodes.jsonl.gz`(403,687 B,gz SHA256 `bcb6b046...6ef51d2`;解压后 JSONL content SHA256
    `cc94661fba940dfc0a5493e38027a4cae18e1872fba9409c17d11d74bddf19d1`)+ `manifest.json`(schema、
    plan 绑定、frozen 协议、生成命令);
  - 目标 repo:`gavinlaw/causalcache-set-utility-variable-history-mobile` 路径
    `artifacts/androidworld-memory-ceiling-v1-76230e4/`;上传后用 immutable revision 链接替换本条;
  - 原始逐文件副本另在 hyper01 `/data02/jaxan/runs/causalcache-ceiling-v3-76230e4/episodes/`;
- 执行:24 emulator + 24 policy workers(8×H200),两轮外部 preflight 清理团灭由 host 侧 supervisor
  自动复活,断点无损;作废的 v1/v2 attempt(suite 绑定 bug)记录在 progress。
