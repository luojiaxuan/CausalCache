# Desktop DiD 三臂 policy training v1 —— 正式 gate 结果

日期:2026-07-26(夜)。语料 `causalcache.desktop_did_sample.v1`(969 组,
train/dev = 773/94,`samples.jsonl` SHA `2ef47af4…b71c1f`);目标 `did_ra_aware`
(交接 §6);三行同协议:DDP4 × accum 4、150 步、checkpoint 每 25 步、
2560 visual tokens、rank 8 / alpha 16。dev = 94 组 / 77 轨迹,
2,000 次 episode-cluster bootstrap;gate 与选点规则在训练前冻结于三份 config
(`b88fb75`)。

## Headline

**HGKV 通过全部预注册 gate,选点 s150:**
`did_select = +0.0111,95% CI [+0.0074, +0.0150]`(ci_low > 0 ✅),
`A_S = +0.0139 > 0`,`|A_r| = 0.0087 < 0.02`,`wrong_drift_abs = 0.0085 < 0.02`。
桌面数据 + DiD 目标下,门控 KV adapter 学到了**内容选择**而不是"见历史图就放大"
—— 旧 GUI-Odyssey 线的两个失败模式(四臂遗漏 RA / legacy SA−R0 假阳性)均未复现。

## 三行对照(各自选点后的终值)

| 行 | 选点 | did_select [CI] | A_S | \|A_r\| (<0.02) | wrong drift (<0.02) | all_must_pass |
|---|---|---|---|---|---|---|
| **HGKV**(last_8 k/v,mask 门控) | **s150** | **+0.0111 [+0.0074,+0.0150]** | +0.0139 | 0.0087 ✅ | 0.0085 ✅ | **✅(s50 起全过)** |
| matched ungated KV(同层位/秩) | s125 | +0.0076 [+0.0048,+0.0103] | +0.0068 | 0.0090 ✅ | 0.0063 ✅ | ✅(s75 起) |
| Full-layer LoRA(36 层 q/k/v/o) | — | +0.0163 [+0.0116,+0.0209](s125,不可选) | +0.0162 | **0.021–0.027 ❌ 全程超 cap** | s50/s150 ❌(至 0.029) | **无合格 checkpoint** |

- **门控净增益(同组配对)**:HGKV@s150 − ungated@s125 的 did_select =
  `+0.00345,95% CI [+0.0007, +0.0062]`,56/94 组占优(组级 bootstrap 10k;
  论文表须换轨迹聚类版,94 组/77 轨迹下差异预计很小)。
- **Full-LoRA 的教训**:它的选择性点估计最高,但漂移全程压不进预注册包络
  (|A_r| 与 wrong drift 双双超 ε=0.02)——"能选择但锚不住";这正是 cap gate
  存在的意义,也反证 HGKV 的价值:**在漂移包络内实现选择性**。
- ungated 对照说明 last_8 k/v 的层位/参数量本身就能学到部分选择性;
  门控在此之上仍有显著净增益。
- 冻结基线:`frozen_selection_effect = +0.0333 [+0.0118, +0.0567]`
  ——桌面冻结模型本就显著受益于目标相关旧帧,现象层前提成立
  (GUI-Odyssey 上因目标 case 稀少无法测得的量)。

## §9 checkpoint gate 清单进度

1. B0 bitwise parity ✅(机械审计,随机非零权重下 max_abs_diff=0.0);
2. `A_S−A_R` dev CI 下界 > 0 ✅(即 did_select);
3. `A_S−A_W` > 0 ✅(did_content 为正,WA gap 见报告);
4. `A_S > 0` ✅;
5. `|A_R|`/`|A_W|` ≤ ε ✅(尾部见报告 percentiles);
6. tool-call parser validity 不低于 Frozen —— **待跑**(生成式评测);
7. Desktop heldout 完整动作等价率不退化 —— **待跑**。
第 6/7 项完成前 **s150 尚未冻结**,selector 标签生成(Stage C)不得启动。

## Artifacts

| 内容 | 位置 | 状态 |
|---|---|---|
| 三份 gate 报告 | 本目录 `gate_report_{hgkv,ungated_kv,full_lora}.json` | Git canonical |
| 训练 checkpoint(各 10 个 .pt) | hyper01 `/data04/jaxan/mw/runs/desktop-did-v1/{hgkv,ungated_kv}/`;hyper00 `/data02/jaxan/runs/desktop-did-v1/full_lora/` | local staging,`PENDING_HF_UPLOAD` → `gavinlaw/causalcache-gui-owl-desktop-memory-adapters` |
| dev 分数缓存 | hyper00 `/data02/jaxan/runs/desktop-did-v1/devscore-*/cache*.jsonl` | local staging |
| 语料 | Hyper01/hyper00 staging(见 corpus manifest) | `PENDING_HF_UPLOAD` → `gavinlaw/causalcache-desktop-memory-training` |

复现:三份 config `code/configs/causalcache_desktop_did_*_v1.json`;
打分 `scripts/score_sparse_history_arms.py`(桌面 schema 支持 `3085296`);
launch 记录见 `docs/progress.md` 2026-07-26(夜)条目。
