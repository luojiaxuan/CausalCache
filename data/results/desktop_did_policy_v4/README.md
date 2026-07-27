# Desktop DiD v4(官方多轮结构重训)—— 运行记录

日期:2026-07-27。背景:Codex 报告确认单轮 renderer 存在协议缺陷(执行器 JSON
历史、像素/[0,999] 量纲混用 —— 实测 31.8% 历史动作 x/y>999、多轮官方结构缺失),
MobileWorld 前例 ef19e1a 表明格式差异可吞掉稀疏选点收益。用户裁定:新建官方
多轮 builder(单轮脚本保留不删),v3 口径整体重训为 v4。

## 语料 v4(fixed-budget replacement,官方多轮 renderer)

- 生成器 [build_desktop_did_corpus_v4.py](../../../code/scripts/build_desktop_did_corpus_v4.py),
  renderer [agentnet_desktop_official.py](../../../code/causalcache/agentnet_desktop_official.py)
  (gap-fold 泛化官方滚动结构;R 臂逐消息退化为官方结构;历史 = `Action:` 行 +
  `<tool_call>`,坐标全 [0,999];保留步缺官方响应整组弃用);
- 组代数与 v3 逐字不变(B∈{1,2,4} 训练、B=8 只建不训、k=1 最老槽位替换、
  age≥B+2、同 seed 轨迹 split);
- 组数:B=1/2/4/8 = 962/756/463/158(v3:969/764/470/160,tripleClick 保留组
  弃用 <1%);train units 1,755;dev 组数与 v3 完全一致(94/72/45/11);
- `samples.jsonl` SHA `e6575f934a6a701c9c50bebfa248dbdf2fc5e244b43d763b0cfe9d3cd96c897c`
  (10,905 行 / 2,181 组);parity_b0 SHA `e33ac3a0…5fd0`;
- 全语料坐标扫描:12,657 行 0 处 tool_call 坐标越界(单轮量纲混用不复现);
- staging:hyper00 `/data02/jaxan/desktop-did-corpus-v4`(权威构建),hyper01
  `/data04/jaxan/mw/desktop-did-corpus-v4`(SHA 校验后镜像,帧 symlink 到 v3 树);
  `PENDING_HF_UPLOAD`。

## 训练三行(16 组/优化步不变式:world × accum = 16)

| 行 | 主机/卡 | world×accum | config | 输出 |
|---|---|---|---|---|
| HGKV | hyper00 GPU0-7(容器 sglang-omni-jaxan) | 8×2 | `causalcache_desktop_did_hgkv_v4.json` | `/data02/jaxan/runs/desktop-did-v4/hgkv` |
| ungated KV | hyper01 GPU0-3(容器 sglang-omni-jaxan) | 4×4 | `causalcache_desktop_did_ungated_kv_v4.json` | `/data04/jaxan/mw/runs/desktop-did-v4/ungated_kv` |
| Full-LoRA | hyper01 GPU6/7(容器 sglang-omni-jaxan-2,待并行会话 MobileWorld 评测释放) | 2×8 | `causalcache_desktop_did_full_lora_v4.json` | `/data04/jaxan/mw/runs/desktop-did-v4/full_lora` |

- 协议冻结面(300 步 / ckpt 50 / lr / margins / gates)三行逐字节一致,单测
  `test_desktop_training_configs_share_the_frozen_protocol[_v4.json]` 只剥离
  accumulation 比较并锁 {2,4,8};
- 监督器:每行 `supervise.sh`(失败全量重启 ≤3 次,冻结分数缓存加速重启;
  退出写 `EXIT_<code>` 标记);已知坑:torchrun 需 `PYTHONPATH=<repo>/code`
  (v4 首启即因此失败 3 次,已修);
- 监控:Mac 侧持久监视器(10 分钟轮询)对 EXIT 标记、checkpoint 落盘、
  train.log 25 分钟无更新(STALE)、probe Traceback、6/7 释放信号告警;
- HGKV 启动确认:training_units=1755,schema v2,config SHA `cae3e984…f3f3`。

## k=1..4 探针(决定是否出 v4.1 加 k≥2 训练臂)

[probe_desktop_official_k.py](../../../code/scripts/probe_desktop_official_k.py)
在 hyper01 GPU0-3 4 分片运行(官方多轮渲染,B=4):R vs S_k(k 个最近 distinct
正例旧帧替换最老 k 槽,age≥6),frozen 边际 + 旧格式 s300(SHA `1754d232…9ff2`)
的 did-style 迁移读数 + 每状态正例可用性分布。输出
`/data04/jaxan/mw/runs/desktop-did-v4/probe-k/probe_k.shard*.jsonl`(断点续跑),
完成标记 `PROBE_DONE_*` 后链式启动 ungated 行。

## 依赖与后续

1. probe 汇总 → 判定 k≥2 是否有收益(有 → v4.1 语料加小比例 k=2 臂);
2. 三行训完 → per-B gate 打分(score_sparse_history_arms,v2 schema 已接线)
   → v4 选点 → §9-6/7 生成式评测(带坐标越界统计)→ selector 标签重打
   (绑定 v4 checkpoint;hyper00 旧格式 singleton 标签 58,376 行保留为对照);
3. 旧格式 v3 结论(B=1/2/4 全过 gate,s300 pooled +0.01106)保留在
   [desktop_did_policy_v3](../desktop_did_policy_v3/README.md),v4 是其官方
   结构复核 —— 若 v4 gate 不过,即桌面版 ef19e1a(收益是单轮结构产物)。
