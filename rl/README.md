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

环境侧(serve 的 `--selector-temperature` / `--rl-audit-dir`、OSWorld worker、VM
基建)留在主仓 —— 它们是共享环境,不属于 RL 目录。

## 迭代闭环

```
serve(τ>0, --rl-audit-dir) × N 副本
  → worker 在 train 切分上 rollout(G=8/任务,cap 30 步)
  → collect_rl_trajectories.py(三键 join,对不上整条弃用)
  → train_causalcache_rl_grpo.py(GRPO + KL 信任域 + 熵)
  → 新 selector_bundle.pt / adapter.pt → server 滚动重启 → 下一迭代
每 20 迭代:held-out 120 任务,τ=0、50 步、2 轮,只报方向。
```

## 状态

- [x] serve RL 模式(主仓 `serve_osworld_official_policy.py`)
- [x] collector / GRPO trainer / ActionScorer / 任务切分
- [ ] `rl_iter_loop.sh` 首跑冒烟(单迭代、2 任务 × G=2)——**下一步**
- [ ] ActionScorer 首跑校验:重建 prompt 的 token 数与 rollout 日志的
      `prompt_tokens` 一致(容差 ±图片 token 化差异),不一致即 fail-closed
