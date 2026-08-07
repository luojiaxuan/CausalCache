# CausalCache-RL(可整体迁出的自包含目录)

冻结 GUI-Owl 8B,RL 联训 selector(Plackett-Luce 随机化)+ HGKV LoRA,
奖励 = OSWorld 任务成败。设计与预注册停止判据见 `docs/rl_pivot_contract.md`。

## 结构(与主仓同构,便于独立成 repo)

```
rl/
  code/
    causalcache_rl/action_scoring.py   # ActionScorer:动作 logprob(梯度过 LoRA)+ KL
    scripts/collect_rl_trajectories.py # result.json + serve 审计 → groups.jsonl
    scripts/train_causalcache_rl_grpo.py
  runscripts/rl_iter_loop.sh           # rollout → collect → train → server 滚动重启
  data/manifests/osworld_rl_{train,heldout}_v1.json  # 241 / 120 切分
  docs/rl_pivot_contract.md
```

## 依赖边界(迁移时唯一要处理的事)

本目录**不复制**主仓代码;运行时 `PYTHONPATH=<主仓>/code:<rl>/code`,引用四处:

| 主仓依赖 | 用途 |
|---|---|
| `causalcache.osworld_official_online.build_official_messages_for_request` | prompt 重建,与 serve **同一个函数**(逐字节同源的保证) |
| `causalcache.policy.gui_owl_v2_vision.verify_frozen_vision_runtime` | 冻结守卫(模型清单 + transformers 5.6.0 源码哈希) |
| `causalcache.policy.history_adapter_context` / `history_token_roles` | 历史 token 掩码与 adapter 作用域 |
| `scripts.train_success_sft_lora` 的 `inject_history_gated_kv` / `lora_state_dict` / `load_lora_state_dict` | LoRA 注入与状态(checkpoint 与 v4/v7 形制互通) |

**server 归属(2026-08-05 变更)**:RL 独占 `code/scripts/serve_osworld_rl_policy.py`
(从主仓 serve fork,含 PL 采样/审计/hidden selector 全部 RL 能力);
主仓 serve 已回滚到 RL 之前,主线评测不受 RL 演化影响。
主仓 bugfix 需要时手动 cherry-pick 进 RL serve,反向永不同步。
OSWorld worker 与 VM 基建仍在主仓(纯环境,双线共用、都不改它)。

## 迭代闭环

```
serve(τ>0, --rl-audit-dir) × N 副本
  → worker 在 train 切分上 rollout(G=8/任务,cap 30 步)
  → collect_rl_trajectories.py(三键 join,对不上整条弃用)
  → train_causalcache_rl_grpo.py(GRPO + KL 信任域 + 熵)
  → 新 selector_bundle.pt / adapter.pt → server 滚动重启 → 下一迭代
每 20 迭代:held-out 120 任务,τ=0、50 步、2 轮,只报方向。
```

## 文档地图

- **方法(怎么做 RL / GRPO 配置 / rollout / 参数更新 / LoRA-not-全参)**:
  `docs/rl_method.md` ← 先读这个
- 实验契约与预注册停止判据:`docs/rl_pivot_contract.md`
- 迭代编排:`runscripts/rl_iter_loop.sh`(h01 宿主侧,阶段 marker 断点续跑)

## 状态

- [x] RL 独占 serve(PL 采样 + 审计 + hidden selector)
- [x] collector / GRPO trainer / ActionScorer / 任务切分
- [x] 冒烟端到端通过(2026-08-05:exit 0,adv=±0.833 精确,零初始化 KL=0)
- [x] `rl_iter_loop.sh` + 可学带任务表(47 任务,6 测量 0<p<1)
- [x] hidden 冒烟端到端(索引遍 2.7s/步;trainer adv 精确对账 0.7035;组内已见真实奖励方差)
- [x] iter-0 bootstrap(hidden 头)+ 首轮迭代已发(ITER 1-4 @ h01,2026-08-05)
- [ ] held-out 评测脚本(τ=0/50 步/2 轮);trainer DDP 分片(提速项,非阻塞)

## 2026-08-05 冒烟交接(容器变更警报)

**两台 hyper 的 `sglang-omni-jaxan` 都被他人重建**(h00 21h 前、h01 12h 前,
/data 挂载已不指向我们的盘)。**h01 的原容器还在,被顶名成 `sglang-omni-jaxan-2`**
(挂载 /data04/jaxan→/bigdata、/data02/jaxan→/data,仓库与模型完好);
h00 的原容器视图丢失,但宿主侧 /data02/jaxan/* 数据完好,需要时按 runscripts/README
第八条重建。
**2026-08-06 更新**:`sglang-omni-jaxan-2` 已按全局规则重建为 `--gpus all`
(原为 device-locked 宿主 4-7;重建前确认容器内无活进程,挂载与镜像不变),
现在容器 GPU 序号 = 宿主序号,下文旧序号换算不再适用。

**冒烟(6 卡内,全部用 `sglang-omni-jaxan-2`)续跑步骤:**
1. bundle 同步 rlsmoke 分支到 /bigdata/osworld/CausalCache(基点取容器内 HEAD);
2. 冒烟 meta(2 个已知翻转任务,libreoffice_writer 0e763496/b21acd93)已生成于
   本机 /tmp/smoke_meta.json,放到 $B/OSWorld/evaluation_examples/smoke_rl.json;
3. server:GPU5,`--memory-budget 2 --selector-temperature 1.0 --rl-audit-dir
   /bigdata/rl-smoke/audit`,端口 19501,容器 IP 端点;
4. G=3 × 2 worker(cap 15 步)→ collect_rl_trajectories(audit + out-g*)→
   train_causalcache_rl_grpo 单卡;
5. 首跑校验:重建 prompt 的 token 数 ≈ rollout 日志 prompt_tokens;
   groups 全 0/全 1 时 trainer 会明确拒绝(选翻转任务就是为避免这个)。

## 2026-08-05 冒烟结果:全链路通到最后一关(OOM),两处已知修法

**已验证通过**:serve PL 采样+审计(71 行,logp 数学自洽)→ 6 episode rollout
(2 任务×G=3,B=2,零报错)→ collector 三键 join(6/6)→ selector 装载 →
HGKV 注入(history_gated_lora)→ prompt 重建(补 official_arguments/full_response
后 build_desktop_official_messages 通过)→ mask([1,seq] 契约 + 目标段补 False)
→ Qwen3-VL 前向(mm_token_type_ids 随目标延长)。

**OOM 根因与修法(下一 session 首件事):**
1. trainer 把整条 episode(12-15 步 × 多图长 prompt)的梯度图攒着才 backward
   —— 改成**逐步 backward**(adv/accum 缩放),接口从 episode_action_logprob
   改为 per-step 回调;KL 与 logprob 合并进同一次带梯度前向,off 侧 no-grad;
2. ActionScorer 没设 visual-tokens —— serve 用 2560,scorer 必须同值
   (看 serve 如何把 --visual-tokens 应用到 processor),否则图片 token 数
   既炸显存又与 rollout 分布不一致。

冒烟 fixture(两退化组并成伪组)只用于走通代码路径,不是方法论。
本轮 6 个修复均已回传本地 rl/ 目录。

## 迭代战报(滚动更新)

| iter | 成功 | 混合组 | KL | 备注 |
|---|---|---|---|---|
| 1 | 11/48 | 4/8 | 0.116 | 首迭代,LoRA 零初始化 |
| 2 | 11/48 | 4/8 | 0.179 | |
| 3 | 18/48 | 7/8 | 0.235 | |
| 4 | 16/47 | 5/8 | 0.225 | KL 趋平,信任域生效;1 ep 重试用尽弃 |
| 5 | 14/46 | 7/8 | 0.244 | band v2 起点 |
| 6 | 9/28 | 4/6 | 0.177 | server 中毒事故:cuDNN mha_graph 逐请求失败,损失 18/48;催生 loop v2 修复遍 |
| 7 | 13/48 | 4/8 | 0.193 | 修复遍生效,48/48 全收(**订正**:曾误记 24/48,那是训练 episode 数不是成功数) |
| 8 | 20/48 | 6/8 | 0.234 | h00 首迭代;断点续跑 credit 35 条 + 补回 13 条 flaky,48/48 全收 |
| 9 | 16/48 | 6/8 | 0.220 | v4 首迭代(8 worker):rollout 64min;serve 并发碎片 OOM ×8,修复遍 18min 自愈,48/48;**首次 DDP train 实跑 ×3 卡 ~13min**(单卡 28-45min) |
| 10 | 18/48 | 3/8 | 0.178 | v5 首迭代:**全程 1h16m**(rollout 67min 零 OOM + DDP 7.5min);退化组偏多属抽签噪声 |
| 11 | 13/48 | 4/8 | 0.189 | 1h24m,零 OOM |
| 12 | 22/48 | 4/8 | 0.224 | 训完即插播 held-out 方向读(eval12.log,RL@12 vs 冻结+recent-B) |
| 13 | 12/48 | 5/8 | 0.253 | |
| 14 | 8/48 | 2/8 | 0.197 | 修复遍 ×1(8 次策略失败) |
| 15 | 18/48 | 4/8 | 0.241 | |
| 16 | 22/48 | 6/8 | 0.252 | |
| 17 | 9/48 | 5/8 | 0.216 | |
| 18 | 10/47 | 2/8 | 0.239 | |
| 19 | 18/48 | 7/8 | 0.197 | |
| 20 | 23/48 | 3/8 | 0.282 | EVAL_DUE → 正式两轮评测(并行,eval20-r1/r2) |

### 20 迭代趋势(训练带,τ=1 探索口径)

11, 11, 18, 16, 14, (9/28), 13, 20, 16, 18, 13, 22, 12, 8, 18, 22, 9, 10, 18, 23

**前十迭代 146/457 = 31.9%,后十迭代 155/479 = 32.4% —— 训练带成功率持平,
20 迭代看不出上行。**KL 全程锁在 0.18-0.28 平台,信任域按设计工作。

如何与 iter-12 方向读的 +6(37.5% vs 32.5%)并置解读,不回避矛盾:

- 训练带这条曲线**本来就不该期待单调上行**:每迭代只抽 8/44 个任务,
  任务难度差是主导方差源(单迭代 95% 区间约 ±13%),而且它是 **τ=1 探索
  口径**——采样本身要付成功率代价;
- held-out 是 **τ=0 部署口径 + 固定 120 任务全配对**,方差小得多,是回答
  "RL 有没有用"的正确工具;
- 两者不矛盾但也**都不构成结论**:方向读单轮 p≈0.12,训练带持平。
  正式两轮评测(iter-20)才是本阶段的裁决;
- 预注册判停(60 迭代无上行则停)**不变**,20 迭代持平不触发任何动作,
  也绝不因此改判据——这正是预注册的意义。

### Reward hacking 全程判读(iters 1-20,rl_task_probe.py)

**零 JUMP、零 FAST6 —— 20 迭代无 hacking 签名。**没有任何任务出现
"成功率跳升 + 成功步数塌缩"(提前终止骗评测器的特征),也没有
"6/6 全成且极快"。绝大多数成功的 med 步数贴着 30 步 cap,符合 OSWorld
终局判分的正常形态。

**9/44 个任务被标 NOISY**(成功率极差 ≥0.5):4c26e3f3、51b11269、
6a33f9b9、6e99a1ad、716a6079、7f52cab9、9f3bb592、aa3a8974、edb61b14。
早先列为观察项的 **4c26e3f3 判定为噪声**(0.17/0.67/0.17/0.33/0.33/0.50),
当时 i5→i8 的跳变是运气不是学习。**不在本 campaign 中途裁带**(换带会让
训练带成功率跨迭代不可比),记为 band v3 候选。

### 选帧—成败耦合诊断(rl_noise_diagnostic.py):一个负结果 + 一次自我纠错

问:NOISY 任务是不是"成败与选帧无关的纯噪声源"?判据:同组内赢家与输家的
选帧统计量之差 Δ(归一化年龄 age、偏离 recent-B 比例 off),组内标签置换检验。

**第一版结论是错的,错在混淆。**全程统计下两组都显著(其他组 age p=0.000、
off p=0.000),看似"非噪声任务的选帧确实驱动成败"。但加对照统计量
`len` 后真相暴露:**赢家 episode 系统性更短**(其他组 52 组中 50 组如此,
平均短 5.8 步)——这近乎同义反复(做对即终止,做错烧满 30 步 cap),
而池子越大越容易选出 recent-B 之外的帧,age/off 的"差异"是长度的机械后果。

**控住混淆后(只看前 10 步,结局分化之前、池子大小可比):**

| 集合 | age Δ / p | off Δ / p |
|---|---|---|
| NOISY(40 组) | +0.019 / 0.038 | +0.007 / 0.610 |
| 其他(52 组) | +0.001 / **0.909** | −0.030 / 0.038 |

其他组的 age 效应从 p=0.000 塌到 **p=0.909**,完全是长度伪影;剩下两项
p=0.038 在 6 次比较下达不到 Bonferroni 阈(0.008)。

**结论(诚实版)**:早期选帧行为**在噪声任务和普通带任务上都无法区分赢家
与输家**。因此:

1. **收回**我此前"GRPO 在 NOISY 任务上学到评测器怪癖"的说法——无证据,
   且 NOISY 与其他任务在这个测度上根本分不开,单独归咎那 9 个任务不成立;
2. 这个负结果适用于**整条可学带**,含义比原假设更重:要么选帧→成败的耦合
   真的很弱(那学习慢就有了解释),要么这两个粗统计量(平均年龄、偏离比例)
   看不见真正起作用的东西(比如"选中的是哪一帧"而非"多老")——
   **无证据 ≠ 证据无**,不可反过来断言选帧没用;
3. 真正的因果检验是 held-out 双臂评测本身(RL 选帧 vs recent-B 固定策略),
   iter-20 正在跑;它比任何观察性统计都干净。

## iter-12 held-out 方向读(2026-08-07,首个闭环判读)

**RL 45/120(37.5%) vs 冻结+recent-B 39/120(32.5%),+6 任务;
翻转任务 12 胜 6 负(2:1)。方向为正,继续训练。**

- 协议:held-out 120(从未参与任何开发环节)× 双臂各 1 轮 × τ=0 部署
  argmax × 50 步 × B=2,同日同机同批任务全配对;**方向 only,按约定不算
  显著性**(12:6/18 翻转在符号检验下 p≈0.12,单轮读数不足以下结论,
  它的职能是"值不值得继续跑到正式评测",答案是值得);
- 中途披露:rl 臂 OOM 缺 21 条时的带偏中读是 +8(12:4),补齐后 +6(12:6)
  ——偏差方向与预判一致(缺的长尾多为 RL 失败案例),诚实修正后方向仍正;
- 胜负任务清单在 `eval-iter12/summary.json`;正式两轮评测按 EVAL_EVERY=20。

**eval12 踩坑全records(供 ITER-20 正式评测)**:
1. worker `--memory-arm` 只有 summary|full,recent-B 是 server 侧默认行为
   (两轮全灭收 0 的真凶);
2. 换臂杀 server 必须等进程退净(垂死进程以 /health 200 骗过健康检查);
3. 臂收 0 条必须 FATAL 不盖章(已修);
4. rl 臂 50 步口径 8-worker 并发 OOM(单次分配 18-36G)——**评测口径并发
   上限 4 worker**,或给 serve 加请求串行锁;
5. 官方 serve 连续服务 ~2h 后显存爬升致小分配 OOM(训练每迭代重启从未暴露)
   ——长评测臂需中途滚动重启或依赖修复遍;
6. 已知瑕疵:修复遍的 PF 判据 grep 的是**累计**日志(跨发射追加),重发后
   必然触发一次空扫修复遍(无害但浪费 ~10min);ITER-20 前改为按本次发射
   增量计数。

- 4 迭代不足以宣称学习信号(任务抽样不同);ITER 20 触发 held-out 评测。
- **band v2**(ITER 5 起):裁掉 30 步 cap 下 ≥12 次测量全败的 3 个任务
  (b21acd93/ce2b64a2/f5c13cdd),44 任务;v1 是 50 步口径估的,cap 更严所致。
  纯调度优化:GRPO 本就跳过全败组,裁掉只省 rollout 不改梯度分布。
- **iter-8 停机事故 ×2(2026-08-06)**:
  1. 他人 `run_multinode.py` 作业占走容器可见 4 卡中的 3 张(各 ~141GB),
     旧 loop 硬编码"server s → 序号 s+1"+ 单次 110s 健康检查直接判死。
     处置:按全局规则重建 h01 容器为 `--gpus all`(重建前确认无活进程;
     此后容器序号 = 宿主序号),loop v3 改 `GPUS` 列表参数化 + 300s 健康
     轮询窗(15s/轮),以 `GPUS="2 5"` 重发。
  2. 重发后 rollout 收 35/48(修复遍未触发,缺失原因待查),trainer 在
     加载模型阶段被**静默 SIGKILL**(无 traceback;非交互 shell 不报 Killed)
     ——同一时刻他人 multinode 作业也集体消失,判断是有人为 benchmark 清场
     整机扫进程,trainer 属误伤。用户随即让机 1 小时。
- **迁移 hyper00(2026-08-06,loop8h00.log)**:用户指示直接去 h00 补跑。
  h00 现容器 `sglang-omni-jaxan` 本就 `--gpus all`+host 网络,挂载
  /data04/jaxan→/data;/data02 满盘(0 可用),工作集整体落 /data04:
  模型/OSWorld/cache-fast/CausalCache(host+容器共用一棵树,bundle 升至
  a53cb66)/run30(只拷 Ubuntu.qcow2,跳过 12G zip)。iter-7/8 状态由
  h01 经 agent 转发 rsync 直传(~1GB,秒级;走 Mac 的 tar 管道实测太慢)。
  loop v3.1:CTN/B/RLH/RLC/REPO/WREPO/MODEL/CACHE 全部 env 化 + host 网络
  CIP 回退 127.0.0.1。iter-8 删 ROLLOUT/COLLECT 标记续跑:35 条既有 episode
  断点跳过,补缺失 13 条。GPUS="3 4 5"(h00 GPU 0-2 为他人占用)。
  **注意:h00 的 rl_iter_8.json meta 是手工放置的**——loop 只在任务采样时
  生成该文件,tasks.json 已存在会跳过;跨机迁移必须手工补。
- **v5 换装(iter-10 起,2026-08-06)**:serve 进程加 `expandable_segments`,
  根治 8-worker 并发下的碎片型 OOM(iter-9 实测 8 次/64min;修复遍虽能自愈
  但每迭代 10-15min 修复税不值)。换装同样走"T 完成即杀-清-重发"脚本
  (relaunch_v5.sh,VM 容器改 ancestor 过滤批量清)。
- **v4 换装(iter-9 起,2026-08-06)**:WORKERS 4→8(8 任务/迭代一人一个,
  rollout 墙钟 ≈ 单 episode)+ train 段 torchrun 3 卡 DDP;预期
  ~1h-1h15m/迭代(iters 1-8 实测 ~2h10m)。换装点选在 iter-8 训完、
  iter-9 rollout 刚起时:杀 loop→SIGTERM worker(留够 VM 自清时间)→
  删 4 个孤儿 VM 容器(happysixd/osworld-docker)→v4 重发,损失仅
  iter-9 前几分钟的在飞 episode。上线确认:health 46s 过,8 worker 起跑。
  **trainer DDP 等价性冒烟(2026-08-06,通过)**:iter-8 原生 groups,
  2 episode × 1 组 × accum 2。前向统计(loss/KL/优势)w1 与 w2 逐位一致;
  param_sum 差异 w1-重跑 8.4e-7(非确定性基线)vs DDP 1.9e-4——后者是
  AdamW 新状态首步 ≈ ±lr·sign(g) 对近零梯度求和序噪声的符号放大
  (~10/660K 参数翻符号),非分片 bug(真 bug 会 1e-2 级且统计对不上);
  rank 间参数逐位一致、episode 认领断言通过。判:等价成立,上线。
  **groups.jsonl 不可跨机复用**:collect 产物内嵌 episode 目录的容器视角
  绝对路径(h01 是 /bigdata/...,h00 是 /data/...),迁移后必须重跑 collect
  (秒级);拿旧机的 groups.jsonl 直接训会在 `_request_from_episode` 处
  StopIteration(DDP 冒烟首发即此坑,误以为是分片 bug)。
  首发三 server 全被冻结守卫拒启:h00 容器被语音项目会话升过包
  (transformers 5.12.1、sglang 0.5.16),守卫钉 5.6.0——**守卫按设计工作,
  这正是它存在的意义**。处置:容器内无活进程,`pip install transformers==5.6.0`
  降回后重发。共享容器跨项目改包环境是常态,迁移到任何"现成"容器都要先
  对钉定版本;语音项目下次在此容器跑若依赖 5.12.1 需自行升回(已知冲突,
  两项目不同时活跃)。
