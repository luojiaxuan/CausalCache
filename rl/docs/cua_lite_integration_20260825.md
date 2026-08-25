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

**判定:任务三(selector/RLOO 进 slime)的 smoke 阶段完成。**
mini-run(train78 × G8 × 30-50 步)就绪,发射前过效率三问
(env 并发↑、每步墙钟压到 <25min、预计总墙钟 12-20h)。

## 5. 同日附加发现(读源/实测拾得)

- 官方折叠把 obs i 的 tool 文本配给 action i 的结论(原版与补丁版同;
  含 obs0 代码路径,真实 episode 不可达);
- 补丁版 CC_DUMP 数据面块引用未定义变量,从未跑通——训练数据面改由
  selector service 决策日志 + shim returns 落盘承担;
- 上游 mobileworld env 支持容器复用(/task/init 快照重载,30 次后回收)
  且 destroy 带 `-v`——RL 吞吐与卷泄漏两个隐患上游已解;
- GUI-Owl-1.5 wire 坐标 ÷999(与 MAI-UI 同族),经 mai_ui 缩放助手复用。
