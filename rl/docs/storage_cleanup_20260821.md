# hyper00/hyper01 存储清理台账(2026-08-21,管理员要求)

## 背景
管理员要求缩减 hyper00/hyper01 上的数据占用。清理前盘点:hyper00 ≈ 2.27T
(/data01 67G + /data02 1.5T + /data04 706G),hyper01 ≈ 2.44T(/data01 394G +
/data02 547G + /data04 1.5T;/data/jaxan 与 /data04/jaxan 同目录不重复计),
合计 ≈ 4.7T。

## 删除纪律
每一项删除满足以下之一,并在主机台账(`hyper00:/data01/jaxan/cleanup_20260821.log`
/ `hyper01:/data04/jaxan/cleanup_20260821.log`)留行:
1. **HF 私有正本已在线验证**(repo 文件数+字节数与本地对照);
2. **公开数据/模型可重拉**(HF hub 公共 repo);
3. **可再生中间物**(特征/token/hidden 分片、下载缓存),且其 manifest/
   receipts/logs 已拷贝到 `kept_metadata_20260821/` 留底;
4. 活跃任务与并行 session 的依赖已逐容器核对(hyper01 的 ASR session 挂载
   venvs/Speech-to-Speech,与被删 GUI 语料无交集)。

## 已执行(第 1-2 批,hyper00 ≈ 890G)
| 路径 | 大小 | 依据 |
|---|---|---|
| /data04 与 /data02 的 agentnet-frames-v3 | 26G×2 | HF `causalcache-agentnet-frames-ubuntu-v3`(25.0GB/8 文件)已验证 |
| /data04/jaxan/hfstage | 26G | 同上传的分片暂存 |
| /data02/jaxan/hf-staging | 19G | HF `causalcache-sparse-history-guiodyssey-v5`(18.8GB/134 文件)已验证 |
| /data01/jaxan/hf_cache | 66G | 纯下载缓存(公开模型) |
| /data04(+02)/datasets 的 infinisst 段/findingdory | ~26G | HF 正本/公开数据 |
| runs/selector-boundary 的 boundary-shards | 270G | 可再生分片,元数据留底 |
| runs/contextual-hidden v3/v4 的 context-shards | ~129G | 同上 |
| artifacts 的 *cache/*tokens/*visual/*partial 8 项 | ~160G | 可再生缓存/半成品暂存 |
| artifacts/models(GUI-Owl-32B/8B、UI-TARS、Qwen3-VL) | ~111G | 公开模型快照 |
| artifacts/sft sparse-v3/v4/v5 | ~77G | v5 有 HF 正本;v3/v4 被 v5 迭代 |

## 已执行(第 1 批,hyper01 ≈ 503G)
| 路径 | 大小 | 依据 |
|---|---|---|
| datasets/agentnet | 374G | 公开语料(xlangai/AgentNet)可重拉;本线衍生物在 HF+tilde |
| agentnet-frames-v3(第三份) | 26G | HF 正本已验证 |
| models/GUI-Owl-32B、8B | ~80G | 公开模型快照 |
| hf-cache | 23G | 纯下载缓存 |

## 未动、待用户裁决(跨 session / 其他项目线)
| 路径 | 大小 | 身份 | 建议 |
|---|---|---|---|
| hyper01:/data04/critic_hack_vla | 494G | critic-hack 线,**今天仍在改** | 该线 session 自行清理;台账先挂账 |
| hyper00:/data04/critic_hack_vla | 44G | 同上 | 同上 |
| hyper01:/data01/video-livr-workspace | 310G | video-livr 线 | 归属确认后再动 |
| hyper00:/data02/autoterm-capacity-zh | 103G | 7 月旧项目(07-11 后未动) | 建议整目录归档或删,需用户确认 |
| hyper00:/data02/RASST_rebuttal | 41G | rebuttal 旧项目(07-13 后未动) | rasst-* HF repo 已存最终产物,建议删 |
| hyper00:/data04/S2S_omni_runs | 60G | MOSS-TTS 运行目录(08-10) | 最终 ckpt 均在 HF(27 个 moss-tts repo),建议删中间物 |
| hyper00:/data02/datasets/chinese_lips | 107G | S2S 输入语料 | 若为公开语料可删,需确认来源 |
| hyper00:/data02/mw + hyper01:/data04/mw | 61G+68G | 移动战役 MobileWorld 基建/运行 | 战役已封存,建议保 receipts 删载荷 |
| hyper00:/data04/v5raw + v5full | 287G+18G | 离线战役 v5 部署评测原始产物(deploy/tokens) | 下一批:细分后删可再生载荷、上传评测 JSONL |
| hyper00:/data04/rl | 33G | 离线战役旧评测 | 同上 |
| hyper01:/data04/osworld | 141G | 旧 OSWorld 基建+运行 | 下一批细分 |
| hyper01:/data01/osworld-v2 + osworld-runner | 42G+38G | OSWorld 基建副本 | 下一批细分 |
| hyper01:/data02/runs + data | 280G+202G | 待细分 | 下一批 |
| hyper01:/data02/models/gui-owl-think | ~17G | **未知自训 ckpt,HF 无正本** | 先上传 HF 再议删 |
| hyper01:/data04/cache | 42G | 可能被活跃容器当 XDG 缓存 | 该 session 结束后删 |

## 保留(活跃)
hyper00:/data04/jaxan/{agentic 37G, osworld 29G, models 17G}(本线 Phase 5/6
正在使用);/data04/jaxan/cache 26G(活跃容器 XDG)。

## 第三批(2026-08-21,用户逐项批准)
video-livr 310G(删前确认零进程引用)、autoterm 103G、RASST_rebuttal 41G、
S2S_omni_runs 60G、chinese_lips 107G、mw 61G+68G —— 合计 ≈ 750G,
依据:用户确认最终产物均在 HF("数据用 hf 当 SoT")。
hyper01 的 `gui-owl-1.5-8b-think` ckpt(HF 无正本)已启动上传
→ `gavinlaw/causalcache-gui-owl-think-8b`,确认落地后删本地。

## 新的长期规则(已写入全局 CLAUDE.md)
1. **单一目录**:hyper00 只用 `/data01/jaxan/`,hyper01 只用 `/data04/jaxan/`;
   其余 `/data*/jaxan` 只出不进,随任务收尾清空;
2. **HF 为数据 SoT**,体积不设限,本地仅工作副本。
现存活跃目录(hyper00:/data04/jaxan/{agentic,osworld,models})在 Phase 5/6
运行结束后迁入 /data01/jaxan。
