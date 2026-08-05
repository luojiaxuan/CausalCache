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
| 7 | 24/48 训 | 4 可用组 | 0.193 | 修复遍生效,48/48 全收 |

- 4 迭代不足以宣称学习信号(任务抽样不同);ITER 20 触发 held-out 评测。
- **band v2**(ITER 5 起):裁掉 30 步 cap 下 ≥12 次测量全败的 3 个任务
  (b21acd93/ce2b64a2/f5c13cdd),44 任务;v1 是 50 步口径估的,cap 更严所致。
  纯调度优化:GRPO 本就跳过全败组,裁掉只省 rollout 不改梯度分布。
- **iter-8 停机事故(2026-08-06)**:他人 `run_multinode.py` 作业占走容器
  可见 4 卡中的 3 张(各 ~141GB),旧 loop 硬编码"server s → 序号 s+1"+
  单次 110s 健康检查直接判死。处置:按全局规则重建容器为 `--gpus all`
  (重建前确认容器内无活进程;此后容器序号 = 宿主序号),loop v3 改为
  `GPUS` 列表参数化 + 300s 健康轮询窗口(15s/轮)。ITER 8–20 以
  `GPUS="2 5" SERVERS=2` 重发(loop8.log);2 server 带 4 worker,
  节奏预计比 3 server 慢 ~20-30%,他人作业释放后可升配。
