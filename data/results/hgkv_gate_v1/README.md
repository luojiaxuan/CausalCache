# History-Gated KV Adapter 正式 gate v1(PASS,s100 冻结)

状态:**GATE_PASS**(2026-07-24)。V1 合同第 10 步完成:checkpoint 只按 GUI-Odyssey heldout
选定,s100 冻结为正式 adapter。

## 判定表(全量 165 heldout 轨迹,5,738 样本 × 2 条件,n=1,220 配对组,10k bootstrap)

| 对照 | s100 适配 | 冻结基线 |
|---|---|---|
| c−b0(历史效用) | **+0.1372 CI[+0.1296,+0.1449] PASS** | +0.0502 CI[+0.0447,+0.0556] |
| c−shuffled(内容辨别) | **+0.0016 CI[+0.0003,+0.0030] PASS** | −0.0142 CI[−0.0171,−0.0112](认证为负) |
| c−irrelevant | **+0.0108 CI[+0.0078,+0.0139] PASS** | −0.0115 CI[−0.0174,−0.0056](认证为负) |
| B0 parity | **漂移 +0.000000 / max_abs 0.000000(n=1,220,逐位精确)** | — |

合同第一阶段唯一问题——"B0 完全不变前提下,正确历史能否在 Odyssey heldout 显著优于
B0/shuffled/irrelevant"——**三条对照 CI 下界全正,答案为是**。冻结模型的内容误用
(两条内容对照 CI 认证为负)被修复并反转。

## 必须如实报告的弱点:wrong-history drift

适配模型把 shuffled/irrelevant 变体的分数也抬高了(相对冻结:+0.0713 / +0.0667;correct
自身抬 +0.0862)。即增益大头是"有历史即放大",内容辨别净边际虽 CI 转正但量级小
(c−shuf +0.0016)。序保持(correct > irrelevant/shuffled,B0 逐位零漂移)判"有界"。
闭环评测若显示错误历史臂受损,此处是第一嫌疑。

## 训练与选点轨迹(12-traj 探针,hyper01)

| step | c−b0 | c−shuf | c−irrel |
|---|---:|---:|---:|
| frozen | +0.0635 | −0.0169 | −0.0121 |
| s25 | +0.0692 | −0.0160 | −0.0118 |
| s50 | +0.1128 | −0.0077 | −0.0004 |
| s75 | +0.1514 | −0.0012 | +0.0104 |
| **s100(峰值,选定)** | **+0.1537** | **−0.0006** | **+0.0099** |
| s125 | +0.1469 | −0.0009 | +0.0095 |
| s150 | +0.1419 | −0.0008 | +0.0116 |

s100 达峰后平台微降;选点仅依 Odyssey 探针 + 全量认证,未触任何 benchmark。

## Provenance

- checkpoint:`hg-s100.pt` = hyper00 `/data02/jaxan/runs/hgkv-formal-v1/lora-step100.pt`,
  sha256 前缀 `8f2cc49e1aa0b06c`;副本 hyper01 `/data02/jaxan/runs/hgkv-eval/hg-s100.pt`;
- 训练:hyper00 6×H200,config `code/configs/causalcache_history_gated_kv_v1.json`
  (rank 8 / alpha 16 / last-8 层 k,v_proj / CE=0 / margins 0.01),数据 ody-sft-v2(37,635 样本,
  heldout 165 轨迹不参训);
- 认证:hyper01 5×H200 14 分片,原始行 `/data02/jaxan/runs/hgkv-eval/cert/`
  (s100-shard*.jsonl / frozen-shard*.jsonl,各 5,738 行),聚合 = pair-group 差 + 10k
  percentile bootstrap;`PENDING_HF_UPLOAD`(逐行 jsonl 待传,阻断原因同前:会话策略禁读 HF token);
- 认证期事故(已修,不影响结论):首次全量图片传输管道中断且解包容器残留覆写 images
  导致 shard4 读到截断 PNG 崩溃;重发时引号 bug + exec 宿主容器退出连坐,第三次独立容器
  重跑补齐。已完成行不受污染(覆写流与正确流内容同源,截断只崩不错分)。

## 下一步(合同第 11-16 步)

dev canary(进行中)→ sealed 零样本 policy 评测矩阵 {Frozen, Full-layer(ody-margin-v2 s75),
History-gated} × {B0, Recent-B, Full history} 于 AW + OSWorld → 过跨平台门后重标 + selector。
