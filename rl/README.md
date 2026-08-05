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
- [ ] iter-0 bootstrap(v4 selector bundle 就位)+ 首轮正式迭代
- [ ] held-out 评测脚本(τ=0/50 步/2 轮);trainer DDP 分片(提速项,非阻塞)

## 2026-08-05 冒烟交接(容器变更警报)

**两台 hyper 的 `sglang-omni-jaxan` 都被他人重建**(h00 21h 前、h01 12h 前,
/data 挂载已不指向我们的盘)。**h01 的原容器还在,被顶名成 `sglang-omni-jaxan-2`**
(挂载 /data04/jaxan→/bigdata、/data02/jaxan→/data,仓库与模型完好);
h00 的原容器视图丢失,但宿主侧 /data02/jaxan/* 数据完好,需要时按 runscripts/README
第八条重建。

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
