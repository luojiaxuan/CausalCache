# CUA-Lite 接入:判分通路验证 ✅ + 满盘事故 + 集成架构(2026-08-25)

接续 commit 59531b3 的三件套待办。本文记录第一件(判分通路验证)的**完成
结论**、当日 /data03 满盘事故的处置、以及第二三件(MobileWorld env 接入
+ selector/RLOO 进 slime)的架构侦察结果与决策。

## 1. ✅ 判分通路验证(P1 旧欠账,已闭环)

**结论:reward 来源已钉死,与我方已对表的官方判分同源。**

### 代码级(读 CUA-Lite@2602509 源码)

* CUA-Lite mobileworld env 的 reward 语义:episode **中途恒 `None`**;
  仅当 `terminate`/`response`(finish 工具)或 `max_steps` 截断时,
  RPC 容器内上游 FastAPI 的 **`/task/eval`** 取 `(score, reason)`,
  score 即 reward(`lite/gym/envs/mobileworld/main.py:845-856`)。
  截断的 episode 也照实评分(与 AndroidWorld 的 terminate-only 不同,
  上游 parity 行为)。eval 抛异常才兜底 0.0,可用 `info["eval_reason"]`
  的有无区分真评与兜底;
* 上游 MobileWorld 官方 runner(`mobile_world/runtime/client.py:274`)打的
  是**同一个** `/task/eval` 端点,`scan_finished_tasks` 只是读 runner 落盘
  的 SCORE_FILE。因此 CUA-Lite 判分 = 官方判分,我方 mw_parity 对表
  (39/115=33.9% vs 官方 37.6%,p=0.341 复现成立)**覆盖 CUA-Lite 通路**;
* 上游源码 pin:CUA-Lite 叠 MobileWorld@8ae5064 源码于官方预构建镜像
  (digest 与本机既有 `ghcr.io/tongyi-mai/mobile_world:latest` 完全一致,
  build 零拉取)。我方对表 checkout(83e7b8f)不含 8ae5064,版本差异
  待港;判分端点与协议两侧相同,暂判无碍,全量前可 diff evaluator 目录。

### 实测(hyper00,fid4/fid5 探针)

| 探针 | 配置 | 结果 |
|---|---|---|
| fid3(旧欠账) | osworld env,1 步 wait,max_steps=2 | reward=None——**按设计正确**(episode 未结束),当时误判为链路断 |
| fid4 | mobileworld,wait→terminate | terminate 被 noop,episode 不终止 → 教训 1 |
| fid5-A | max_steps=1,截断路径 | **reward=0.0 (float), truncated=True, eval_reason="No email found"** |
| fid5-B | extra_tools=["terminate"],终止路径 | **reward=0.0 (float), terminated=True, eval_reason 同上** |

`eval_reason="No email found"` 是评测器真实读了 Mail 后端状态给出的判词
(AcceptMeetingTask 要求回复 Daniel 邮件,探针什么都没做,0 分正确),
证明走的是真 `/task/eval` 而非异常兜底。

**教训 1(会咬人的默认值)**:mobileworld `configs/default.yaml` 默认
`extra_tools: []` —— `terminate`/`response` **不注册**,agent 发终止指令
会被当未知工具 noop,episode 跑满 50 步。**RL 配置必须显式
`extra_tools: [terminate, response]`**(官方 agent 工具面含二者)。

**残留风险(低)**:探针只见过 reward=0.0,未见 1.0。评分函数与对表
一致 + eval_reason 真实,判 1.0 通路同源无虞;smoke 验收第 1 条
(reward 非全 0)会用真 agent 闭环覆盖。

## 2. ★ 事故:/data03(docker 根)100% 满盘

### 时间线(本地时,2026-08-25)

* ~10:30 前:/data03 3.5T 用满,**0 可用**。dockerd 10:43 经历一次重启;
* 受害:①我方 mobileworld 镜像 build 在 apt 步骤报
  "At least one invalid signature"(gpg 写盘失败的表象,起初误导向网络);
  ② **接力链 B11 段 10:57-11:02 空跑,judged=0**——runner 起步即在对
  mw_random 容器 port 6821 的 suite 切换上 300s 超时;③他人容器
  sglang-omni-wenyao-bench 同窗口 Exited(128);
* ~11:15:约 500GB 被释放(并行 session 清理或 dockerd 重启回收,未考证),
  盘回到 87%;
* 11:30-12:00:本 session 完成取证与清理,盘回到 **71%,985G 可用**。

### 根因与归属(取证后确认)

`docker system df`:**6303 个卷、账面 209.5TB、99% 可回收**。对全部
6260 个 dangling 卷做内容取证(helper 容器只读挂载,按内容签名分类):

| 类别 | 数量 | 签名 | 归属 |
|---|---|---|---|
| OSW_VM | **6077** | data.img/uefi.rom/qemu.mac(qemu 盘) | 我方 OSWorld VM 容器 churn |
| MW_DIND | **183** | overlay2/image + 内层 repositories.json 含 mattermost/mastodon/mall | 我方 MobileWorld 容器 churn |
| 未知/其他 | **0** | — | — |

**100% 归我方两条基准线**:历史池脚本 `docker rm` 未带 `-v`,每次容器
churn 都留一个匿名卷。这是收尾清单的系统性欠账(2026-08-21 清理线只扫了
文件盘,没扫 docker 卷)。

### 处置(删除纪律记录)

1. build cache prune(24.9GB 账面,可再生,主要为我方当日两次 build);
2. **删除全部 6260 个已归因 dangling 卷**(`docker volume rm` 批量,
   in-use 卷天然拒绝,误伤面为零)。卷内容均为可再生运行态(qemu 系统盘
   overlay / DinD 内层 docker),实验结果早已导出 /data01 并进 Git/HF,
   无唯一数据。分类清单留档 hyper00:`/data01/jaxan/vol_classify.tsv`;
3. 不动:其他租户 30 个 exited 容器(全部他人)、任何 active 卷。

**规则修正(向前)**:凡删除 emulator/VM 容器一律 `docker rm -f -v`;
池管理脚本同改。CUA-Lite 自身的 destroy 已是 `rm -f -v`(上游内置),
用它的容器生命周期不会复发。

### B11 判读与边界

B11(recent-B11 全量 117,外审决策表的静态证据)**当日产出无效
(judged=0),需要重跑**。但 mw_random 池 + GPU6 v6 vLLM 正被并行
session 的 `p05_multiseed.sh`(多种子扫,当前 r3 轮)占用——探针线归
该 session,**本 session 不碰、不重跑 B11**,只在此记录:盘已修复,
B11 重跑条件已具备,multiseed 收官后由持链方补。

## 3. 集成架构(任务二三的侦察结论与决策)

### 上游已有的(直接用)

* **mobileworld env 完整存在**:161 任务注册(117 GUI-only + 44
  agent-user-interaction)、官方判分(§1)、容器复用
  (`/task/init` 快照重载,`max_resets_per_container=30` 后回收)、
  `ask_user` 走 USER_AGENT_* 转发(=我方 172.17.0.1 修复的上游等价物)、
  destroy 带 `-v`。**镜像已在 hyper00 build 好**
  (`cua-lite/mobileworld:latest`,基座零拉取);
* **GRPO 管线交钥匙**:`scripts/train/run_grpo.sh`(Ray+slime+Megatron+
  SGLang,权重热换 slime 原生),`ROLLOUT_MODULE` 支持 out-of-tree recipe
  (examples/grounding 为模板),`CUA_LITE_REGISTRATION_MODULES` 注册
  自定义 agent;`mixed_group_ratio`/`nonzero_return_rate` 等 v2 监控项
  上游已内置(`lite/train/rollout/grpo.py`);
* **执行器架构免移植**:GUI-Owl-1.5-8B config =
  `Qwen3VLForConditionalGeneration`(model_type qwen3_vl),CUA-Lite 的
  qwen3_vl Megatron bridge / model args 直接可用;
* slime 估计量:grpo/gspo/reinforce++(_baseline)/ppo。executor 用 grpo
  (组内基线+PPO 裁剪,合 v2);selector 的 RLOO 在我方侧车实现(见下)。

### 必须新建的(工作量主体)

1. **`gui_owl` agent family**(消息构造 parity 是硬约束):CUA-Lite 现有
   `mai_ui` family 与官方 `gui_owl_1_5.py` **不同构**——官方把折叠文本
   步骤揉进首条 user 文本(`WITH_HISTSTEPS` 模板)且首图挂在首条 user,
   mai_ui 则 instruction 单独成泡 + 旧 assistant 原文成串。直接用 mai_ui
   会同时改变输入分布和丢掉 33.9% 对表锚点。做法:新 family 复刻打过
   补丁的官方构造(S 参数化 + learned 分支 + CC_DUMP),88 用例 parity
   fixture 平移为字节级回归测试;
2. **selector 侧车**:`rl/grpo/selector.py`(PL 头)+ `selector_service.py`
   (rollout 期推理服务,决策与特征落盘)基本原样复用;训练侧新写
   **selector RLOO trainer**(独立轻量进程):消费 rollout dumps
   (S、PL logp、特征)+ episode returns(由 recipe 的
   convert_samples_to_train_data 旁路落盘),算 RLOO 优势 + PL 联合比率
   裁剪 + 归一化熵,步进 41M 头,热推权重回 service。slime 承载 executor
   全参 PG,二者共享同一批 episode 奖励 = joint 训练,Megatron 零手术;
3. **recipe 包**(放本仓 `rl/cua/`,out-of-tree,CUA-Lite 上游钉
   2602509 不 fork):registration(gui_owl family + mobileworld learned
   变体 config)+ ROLLOUT_MODULE shim + 任务导出(117 GUI-only 按 app
   分层 train78/heldout39 parquet)+ 发射脚本(env-server + selector
   service + run_grpo.sh 编排)。

### 决策日志(默认推进,可推翻)

| 问题 | 默认choice | 理由 | 回滚 |
|---|---|---|---|
| 6260 孤儿卷删否 | 删(已执行) | 内容签名 100% 归我方;无唯一数据;共享盘满是最高级违规 | 不可逆但无损:卷为可再生运行态;清单留档 vol_classify.tsv |
| B11 judged=0 重跑归谁 | 不动,归持链 session | mw_random 池被 multiseed r3 占用;双驱同池必撞 | multiseed 收官后任一 session 皆可按 chain_b11.sh 段 [2][3] 重发 |
| selector 训练进 slime 内核还是侧车 | 侧车(独立进程) | Megatron 零手术;train_smoke.py 的 RLOO/PL 裁剪数学直接复用;32×H20 交接同形 | 若全量阶段需要 per-step critic 或更紧耦合,再评估进 slime loss 内核 |
| executor 估计量 | slime `grpo` | 组内基线+裁剪合 v2;slime 无现成 rloo | 需要严格 RLOO(不除 std)时在 slime 加估计量,改动很小 |
| agent family 落点 | 本仓 out-of-tree recipe | 上游不 fork,升级面窄;run_grpo.sh 原生支持 | 若上游接受,可回流 PR |

## 4. 下一步(顺序)

1. ✅ gui_owl family(同日完成):adapter/action_space/protocol/agent
   四件套,**204/204 字节级 parity**(fixture 由官方 venv 内 monkeypatch
   LLM client 生成,306 step-case,askuser 案例因 env note 格式偏差跳过);
2. ✅ selector service/trainer 侧车 + recipe 包 + 冻结切分(78/39)+
   任务导出脚本;
3. ✅ e2e 单 episode(真环境+真 GUI-Owl-8B vLLM):8 步做完
   AcceptMeetingTask,**官方评测器判 reward=1.0**——任务 1 的残留风险
   (未见过非零 reward)就此闭合;CC_TRACE 窗口滑动与官方语义一致;
4. 🔴 slime 容器 smoke(四条验收)→ mini-run(train78 × G8 × 30-50 步)
   → 32×H20 runbook。发射清单见 rl/cua/README.md。

## 4.5 smoke 基建进度与决策(滚动)

- ✅ slime 子模块已拉(SSH url 改 https);slimerl/slime:v0.3.0 拉取中;
- ✅ 标点集 + add_period_robustly 与官方 helpers **程序化比对全等**
  (punct_check.py,防手抄漂移);
- GPU 计划:hyper00 GPU 0/1/2 空闲 → rollout 1 卡 + train 2 卡(ASYNC);
  GPU6 探针 vLLM 与 mw_random 池归并行 session,勿动;
- **决策:slime 镜像重打为 `jaxanluo/sglang-omni:trainer`**(而非字面
  白名单 `sglang-omni:dev`——该名已被他人 `hongccc/sglang-omni:dev` 族
  占用语义,且我方 `jaxanluo/sglang-omni:dev` 仍指向旧镜像不宜重指;
  `<owner>/sglang-omni:<tag>` 是本机同族既有形态,混同意图达成,零覆盖)。
  容器名照规范 `sglang-omni-jaxan-<N>`(自复刻 docker run,launch.sh 的
  名字/镜像硬编码不适用);补挂 cc_recipe、/data04/jaxan models、pyshim;
- mw_rl 池 8 台:CUA-Lite 自管容器用不上,**smoke 通过后删除**(暂留作
  路线 B 兜底:若 slime 通路遇墙,官方 runner + 手搓迭代仍可用它跑)。

## 4.6 ★ smoke 终判(2026-08-25 深夜,第六发):**四条验收全过**

| 验收 | 证据 |
|---|---|
| ① reward 非全 0 | 第六发批2 `nonzero_return_rate=0.12`(4/32 成功、2 混合组);第四发批1 0.09;e2e 单集 reward=1.0 |
| ② 选帧分布在变 | 行为层:805+ 条 CC_TRACE,S 非连续多样(如 [6,15,21]、[15,23,28]);参数层:selector 首轮真实训练 391 步后 `rel_drift=0.00275`,`/reload` 热换生效,后续决策由新参数驱动 |
| ③ loss 有限 | selector loss 0.048 / grad 0.0037;executor step1 loss 有限、`grad_norm=2.03`、`nan_inf_count=0`、`adv_abs_max=9.16`(优势真实流动) |
| ④ 权重真热换 | selector `/reload=true`;executor `update_weights` 实测 1.76s 完成,批3 rollout 使用更新后权重 |

管线形态(实测):32 rollout/批全收零损耗;批间 `step_time≈38min`
(rollout-bound,`wait_ratio=0.71`,train 11min/批 @TP2+CPU-offload,
136 TFLOPS/卡)。**mini-run 前必办的两件调优**:提高 env 并发
(16→32+,CPU 空间充足)压 rollout 墙钟;训推失配
`train_rollout_logprob_abs_diff=0.034` nats(bf16 双引擎正常带内)
列入监控,涨破 ~0.1 需查(v2 监控清单第 5 条)。

### smoke 事故账(8 起,全修复入库)

| # | 死点 | 根因 | 修法(commit) |
|---|---|---|---|
| 1/2 | worker 首连 raylet SETTINGS 超时(100% 复现) | 容器 nofile soft=1024,Ray 按 224 核 prestart 吃满 FD,NM accept 冻结 | `--ulimit nofile=524288`(9e562e6) |
| 3 | NCCL NVLS CUDA 401 | 共享 H200 Fabric Manager 不支持 NVLS,上游 nvlink.sh 探到即强开无覆盖口 | worktree 补丁尊重预设 + `HAS_NVLINK=0`(ee4cb0b) |
| 4 | selector 服务 500(哑) | step0 空帧 torch.stack 崩;500 无日志难定位 | 双侧修+异常落日志(13c27f7) |
| 5 | "turn 无图像" 炸 episode | env 截图瞬时失败产生无图 turn(官方 runner 无此形态) | S 候选限带图 turn+当前帧回退,204 parity 复跑全绿(e368b30) |
| 6 | returns 恒空(fail-loud 捕获) | adapter 写 env.metadata 是错误通道(segmenter 读 slime 侧样本 metadata);静默 except 掩盖 | engine.py 补丁注入 cc_episode(34e42bb) |
| 7 | 第五发一小时静死 | pkill 匹配自身 ssh 命令行自杀;job 死于脏 session 预检而监控模式没抓预检类失败 | 脚本文件化+net env-server 重启+宽监控模式 |
| 8 | selector 反传崩+daemon 静死 | PL slate_logprob 原地 mask 改写毁 autograd 版本(手搓 smoke 从未真正反传过,潜伏 bug) | mask 逐步 clone+daemon 兜异常保活(bb3abce) |
| 9 | B11 v2 秒退+残留误读 | **匿名化反噬**:官方 runner `discover_backends` 按镜像子串 `mobile_world` 过滤,池镜像改 `sglang-omni:env0` 后 0 台可见;退出后 v1 残留 22 个快任务被归约读成 86.4%(幸存者偏差,真实 recent 口径 ~31%) | v3 改 `--aw-host` 显式 URL 列表跳过自动发现;**规则:改镜像/容器名前先 grep 按名发现的消费方,改完立刻验证发现仍通** |

**判定:任务三(selector/RLOO 进 slime)的 smoke 阶段完成。**
mini-run(train78 × G8 × 30-50 步)就绪,发射前过效率三问
(env 并发↑、每步墙钟压到 <25min、预计总墙钟 12-20h)。

## 4.7 ★ recent-B11 终判(2026-08-25 深夜)——外审决策表最有利的一格

**终值**:`judged=69/117, success=42, rate=60.9%`;48 题因 **32k 上下文
装不下 12 图历史**未能完成判分(溢出相关日志 137 行)。

**配对比较(同 69 题,消子集偏差;B2 三个独立轮)**:

| 对照轮 | B2 | B11 | Δ | 失配对(仅B11胜:仅B2胜) |
|---|---|---|---|---|
| recentB2(r1) | 43.5% | 60.9% | **+17.4pp** | 14:2 |
| recentB2_r2 | 42.0% | 60.9% | **+18.9pp** | 14:1 |
| recentB2_r3 | 44.9% | 60.9% | **+16.0pp** | 13:2 |

符号检验 p≈0.001-0.004,三轮方向一致。**判读(外审决策表)**:
可行子集上 **B11 ≫ B2**(视觉历史有大额增益)+ **41% 任务全历史在
实用上下文预算内不可行** —— 同时给出稀疏选择的**靶子**(用 B=2 便宜
找回全历史增益)与**必要性**(全历史装不下)。这正是 Option A 最强的
证据格。限定:69 题子集偏短偏易(B2 在其上 43-45% vs 全量 ~30%),
Δ 只声明于该子集;溢出题上 B11 实际不可用,属可行性论证而非不公平。

**B11 事故账**:v1 被满盘事故打断(残留 22 题曾被误读 86%,幸存者
偏差);v2 匿名化反噬(官方 runner 按镜像子串发现后端,池改名后 0 台
可见秒退);v3 显式 `--aw-host` 成功,全程 6.5h(12 图 prompt 单卡
vLLM 是瓶颈)。收尾:v6 vLLM 已删(GPU6 释放),池 32 台 KEEP(map
注明:mini/全量 recent 对照与 cross-play 评测用,全量收官时删)。

## 4.8 ★ max-fit 合成臂终判(2026-08-26 凌晨)——"堆满 history"两头都不是答案

**动机**(用户提议):对 B11 溢出的 48 题逐级退 B=10→8→6,取每题最大
可行 B,合成一条"近似全 history 上限"臂,补齐 4.7 缺失的 41% 覆盖。

**执行**:B=10 一级即全部装下(降级链 8/6 零触发);vLLM 临时实例
(GPU7,与用户授权混卡的 util=0 占位同卡)用后即删,map 已清。
`ThanksgivingPrepTask` 多次重试未能完成判分(最长任务之一),按缺失记,
故合成臂分母 116=69(B11)+47(B10)。逐题结果落
`/data01/jaxan/sglang-omni-rl/maxfit_arm.json`。

**终值与配对**:

| 口径 | max-fit | recent-B2(三轮) | 判读 |
|---|---|---|---|
| 全量 116 题 | **37.9%**(44/116) | 31.0 / 28.4 / 30.2% | 合成臂整体仍胜(可行 69 题的大额增益主导) |
| 溢出 47 题 | **4.3%**(2/47,B=10) | **12.8 / 8.5 / 8.5%**(均值 9.9%) | **B2 反超一倍,三轮方向一致** |

溢出子集上 maxfit 仅解 ItemCheckoutTask、MastodonMallPurchaseCommodityTask
两题;B2 任一轮合计解出 6 题,其中 5 题 maxfit 全败(CartManagement、
CheckGithubInfo、MastodonImportMutedUsers、MastodonReply、
MattermostIncidentEscalation)。限定:B10 仅单轮,2 vs 4-6 的计数差
单独不显著;但三轮 B2 全部 ≥ 2×,方向可信。

**判读(与 4.7 合并成完整故事)**:短任务上 B11 ≫ B2(+16-19pp),
长任务上 B=10 把 32k 上下文塞到贴顶反而**劣于 B=2**——机制上无论是
注意力稀释还是完成空间被挤占,结论一致:**"尽量塞满历史"在两头都不是
最优,固定小 B + 学习选哪几帧是唯一能同时吃两头的设计**。Option A 的
动机从"必要性"(装不下)升级为"充分性"(装下了也更差)。此表为外审
决策表新增最锋利的一格;全量训练收益预期也应据此校正:selector 的
上限不是 60.9%(那只是可行子集),而是"短任务追 B11 + 长任务不劣于
B2"的合成线。

## 4.9 ★ mini-run v5 终判(2026-08-26)——管线全绿,学习判读如预期"测不出而非失败"

**收官**:30/30 批(恢复前 12 + 恢复后 18),`Job succeeded` 06:19;
全程含 2 次 OOM 事故(→900g + lazy-expand 后稳定,峰值 249G/900G)与
多次 ASYNC 训练步窗口的 503 熔断(定性为固有形态,重试自动扛过,零损失)。
证据小件(returns/task_stats/selector 曲线与权重)入库
`rl/results/mini_v5/`。

**四条判读**:

1. **executor:30 批测不出效应(如预期)**。批均奖励三分段
   19.3%→28.6%→37.4% 看似陡升,但逐任务配对(首尾段皆有样本的 15 任务)
   4 升 5 降 6 平、平均 Δ=+0.3pp≈零——**上升全部来自难度优先采样器把
   预算移向可学任务(设计行为),不是策略变强**。mini 的角色本就是
   管线验证+统计收集,效应检测需要全量的批规模。
2. **selector:权重在动、行为未动 → lr 结论坐实**。930 步 trainer,
   相对初始漂移终值 3.2%,但 mean_frame_age 全程 2.45 不变、sel_loss
   平在 0.0111。**全量 selector lr 1e-4→1e-3**(方向确定,幅度=TODO
   的首个消融),并加"行为位移"监控(frame_age 分布、与 recency 的
   KL)而非只看权重漂移。
3. **难度优先采样先验落盘**:58/78 模板被采,总组 140、混合组率 41%;
   采样正确偏向可学任务(SharePhotos 7 组、CheckDeduplicatedEvents
   5 组=4 混合)。18 个任务 ≥2 组仍零混合(当前策略下死信号,靠 25%
   uniform floor 维持探索)。`task_stats.json` 作为全量 warm-start 先验。
4. **高分抽检合格**:末段 132 个成功 episode 集中在体面任务
   (ScheduleCoffeeTimeViaSms/MastodonNewPost/ChangeHeader 等),
   成功轨迹步数 11~49、中位 29~44,无 1~2 步通关的 hacking 签名。

**收尾处置(已执行)**:侧车按 PID 文件停止、Ray 停止,GPU 1/2 清零;
GPU 0 上随后出现的 110G 进程经 cgroup 核属为 chenye 容器(我方结束后
上卡的正常共享使用,未动)——教训重申:kill 前先核归属,容器内盲杀
循环因 PID 命名空间隔离才碰巧无害。checkpoint 清理:删 Megatron dist
分片(66G+98G)与中间 hf 导出,仅留终版 `hf_p2/iter_17`(17G),
释放 ~313G(盘 73%);正本=选择器/指标/returns 已入 Git(2d21bcf),
iter_11 在 HF。容器与池 32 台 KEEP(map 已注明:待全量发射,取消则删)。

## 4.10 ★ 全量发射事故账(2026-08-26 夜,六发才通)+ 在轨基线

外审(fullrun_launch_review_20260826)后发射。六连发事故与修复,
**共同根因:发射脚本基于 smoke 期 repo 版起笔,未对照宿主演化版
mini_resume_v5.sh(其早已含四件必设)**:

| # | 症状 | 根因 | 修复 |
|---|---|---|---|
| 1 | NVLS CUDA 401,NCCL 组网即死 | 漏 `HAS_NVLINK=0` | 补 env(smoke #3 原样复发) |
| 2 | 30 分钟零完成,stall 看门狗全批取消 | ENV_CONCURRENCY 48 击穿单引擎(排队使单步 46s) | 降 32 + `ROLLOUT_STALL_TIMEOUT_S=3600` |
| 3 | 批 1 train 步 GPU OOM(词表 logit 5.58G 挤不进) | 漏 `TP_SIZE=2` + `OPTIM_CPU_OFFLOAD=1`,Adam fp32 驻 GPU | 补 env,对齐 mini 全套 |
| 4 | 新发射被拒:dirty session | 前发死亡时 24 实例挂 in-flight | env server 换新+清容器 |
| 5 | 再拒:cleanup 后仍有 live 实例 | **pid 文件指向重启失败的新实例,原始 env server 一直霸占端口**带僵尸注册表 | 按真身 PID 处决,验证 `{"instances":[]}` 再发 |
| 6 | selector trainer 消费 188 条后报表异常 | **跨发射 group_index 碰撞**:第三发 60 条残留与批 1 的 64 条同组混算 RLOO 基线;同时发现 trainer 默认 service-url(127.0.0.1)与 service 绑定(172.17.0.1)不匹配,**reload 一直静默失败** | 归档污染数据面+重置 selector=S₀+trainer 显式 `--service-url`;脚本加"重发前轮转 returns/decisions"纪律 |

事故 6 的两个幸运与一个复核:reload 失败反而隔离了污染(错误权重
从未达服务端,批 1-2 rollout 实际全程 S₀,干净);探针库指标在污染轮
实证全链路可产出(probe_kl_s0=0.052、jaccard=1.0、recency_mass=0.574、
clip_frac=0.368);**mini 审计:35 轮 reload:true 全成功,不受此 bug
影响,§4.9 的 lr 结论维持原判**。

**拓扑演化(当日用户两次授权)**:3 卡(train 1/2 + 引擎在混卡 GPU5)
→ 用户授权 GPU3 混用 → 四卡双引擎(train TP2=GPU1/2,引擎=GPU5/3,
MEM_FRACTION 0.55)→ 用户点名授权 kill GPU3 的 45.8G 泄漏 worker
(chenye 容器内 PID 690418,2 天龄、5 秒连续 0% util;仅杀该进程,
其 GPU0 活任务未动)→ GPU3 全净。

**在轨基线(前两批)**:批墙钟 35/39 分钟(单引擎版 56)、64/64 全收、
混合组率 62%/38%、rate 0.31/0.16、批 1 权重热换 1.9s、容器内存 348G
稳态(offload 生效)。预计 100 批 ≈ 2.6 天。selector 数据面自批 3 起
干净(损失前 2 批 selector 样本,executor 零影响)。

## 4.11 全量在轨监控发现(2026-08-27,前 21 批)

**QA 记忆化通道(高分抽检纪律的直接产出)**:CheckDeduplicatedEvents
成功率 3/8→12/21(57%)且全为 1 次选帧调用的 ~3 步速解。读源实锤:
该任务 goal 是"数日历事件数",评测器做 `correct_answer in str(answer)`
**文本子串匹配**;训练集是冻结实例 → 正确数字恒定 → **RL 可直接记忆
答案数字刷分,不需要真看日历**。枚举:train78 中 **12 个任务(15%)**
走 interaction_cache 文本匹配评测(CheckCartPrice / CheckConference
Duration / CheckDeduplicatedEvents / CheckInvoice1 / ChromeSearch
BeijingWeather / CountFileLines / GoogleMapsAlibabaSouthNeighbor /
ReadQwen3Paper2/3/4 / RecentTotalExpense / SendForms)。**汇报纪律
(即日生效)**:训练奖励与评测一律分两列报——态验证 66 题 vs 答案
验证 12 题;heldout-39 是唯一真话;12 题名单的逐任务奖励曲线跟踪
记忆化签名(快速爬向全对+超短解)。不改任务集(冻结契约),
这是测量口径修正,不是管线修理。

**selector 小批聚合改造(预登记判据命中)**:clip_frac 三连
0.68/0.71/0.75(>0.5 线),probe_recency_mass 折返(0.385→0.548)
呈振荡不收敛——逐条 opt.step 的轮内千步漂移把大半比率推出裁剪域,
正优势更新失效。已改:决策全收集+洗牌+64 条 minibatch 均值步进
(轮内漂移 ~64 倍降),trainer 热替换部署,主跑零中断。判读:后续
clip_frac 应显著回落;recency_mass 曲线转向单调即为收敛信号。

**其余在轨读数**:前 21 批节奏 ~36 分/批,混合组率带 25-75%(均值
~50%,优先采样维持信号密度),InstanceGone 稳态 ~9%/批(conc64 批
边界税),内存 307-348G 稳态,checkpoint 链 hf_p2 iter 2..17 完整,
第 5 保存点高分抽检无 JUMP/FAST6 签名(QA 通道单列如上)。

## 5. 同日附加发现(读源/实测拾得)

- 官方折叠把 obs i 的 tool 文本配给 action i 的结论(原版与补丁版同;
  含 obs0 代码路径,真实 episode 不可达);
- 补丁版 CC_DUMP 数据面块引用未定义变量,从未跑通——训练数据面改由
  selector service 决策日志 + shim returns 落盘承担;
- 上游 mobileworld env 支持容器复用(/task/init 快照重载,30 次后回收)
  且 destroy 带 `-v`——RL 吞吐与卷泄漏两个隐患上游已解;
- GUI-Owl-1.5 wire 坐标 ÷999(与 MAI-UI 同族),经 mai_ui 缩放助手复用。
