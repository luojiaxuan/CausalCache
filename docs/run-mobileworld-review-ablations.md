# Run 记录:review 响应四臂(text-only / OCR / 32B 跨骨干)

日期:2026-07-28 夜。响应外部 review 的 P0 项:像素-响应混杂隔离、OCR 强
baseline、同家族更大骨干。全部 MobileWorld 117 任务单轮、B=4。

## 臂定义

| campaign | 服务器 | 目的 |
| --- | --- | --- |
| mw-cc-textonly-v1 | 8B+HGKV+selector,`--history-render text_only`(58253,GPU2) | 保留轮只恢复 verbatim 响应、不给图 → 隔离像素贡献 |
| mw-cc-ocr-v1 | 同上但 `--history-render ocr`(58477,GPU3) | 历史图 → tesseract 5.3.4 OCR 文本 → 像素 vs OCR 文本 |
| mw-32b-cc-v1 | GUI-Owl-1.5-32B 冻结,无 HGKV,+selector 两遍(58141,GPU2) | 跨骨干:selector-only 迁移 |
| mw-32b-recent-v1 | GUI-Owl-1.5-32B 冻结,Recent-4(58365,GPU3) | 32B 同预算 recent 参照 |

- `--history-render` 变换在 pass-1/pass-2 生成前统一应用:除最后一张
  (current 截图)外的所有 image 项删除(text_only)或替换为
  `[restored screenshot, OCR text] …`(ocr);空 content 兜底
  `[screenshot omitted]`。selector 的特征计算不受渲染影响(仍看完整数据),
  只有 policy 看到的 prompt 变化——这正是要隔离的变量。
- 32B:`mPLUG/GUI-Owl-1.5-32B-Instruct` rev `6154cc22…21ac`,
  snapshot manifest 25 文件逐一 sha256
  (`gui_owl_1_5_32b_snapshot.json`);无 adapter(HGKV 是 8B 专属,
  此对照按 review 建议做 selector-only 跨骨干)。configs:
  `causalcache_mobileworld_official_b4_32b_{sel_,}v1.json`。

## 拓扑与容量

- hyper00 仅 GPU2/3 可用(0/1 他人 idle-hold 进程、4-7 他人满载)。
  共置:GPU2 = 32B-CC + 8B-textonly;GPU3 = 32B-recent + 8B-OCR
  (text/OCR prompt 无历史图,轻)。
- 48 模拟器(mwh-/mwh2-/mwh3- 三舰队并行 boot)→ 4 campaign × 12
  (4 shard × 3 envs)。
- 教训+1:手搓 fleet-shard manifest 必须保留原 manifest 全部顶层字段
  (image/source_revision 等),runner 有冻结校验
  "fleet image differs from frozen config";首启因此 STALL,保字段重切
  后恢复。

## 存活

- supervisor 断点续跑 + 心跳同剂量扫描;本机持久 Monitor 7 分钟一轮。
- 预期:12 模拟器/臂,单轮 117 任务约 8-10h;6h 检查点报部分结果
  (memory-critical 62 子集与全量同步累积)。

## 更新(2026-07-28 深夜):32B 降级、双机扩容、两个新臂

- **32B 两臂撤下**(用户裁定非 P0:32B 需重训 HGKV/policy 才有意义,
  见任务 #2)。已跑的 32B 服务器/监督器全部 kill,资源让给 text/OCR。
- **双机扩容**:h01 恢复后以 count=8 全局分区加入——h00 跑 shard 0-3
  (`/data/mw/runs`),h01 跑 shard 4-7(`/bigdata/mw/runs`),两臂各
  48 模拟器。h01 canonical 容器只映射 GPU 0-3,GPU 4-7 由辅容器
  `sglang-omni-jaxan-2`(172.17.0.55)承载 server;h00 GPU2 中途被
  外部 106GB 进程抢占,OCR#2(58478)移至 GPU3 恢复。
- **新臂准备(零延迟翻臂)**:text/OCR 两臂在某机全部 shard DONE
  (心跳 `pending_at_attempt_start==0`)后,本地触发器立即在该机执行
  `flip_h0{0,1}.sh`:清场 → 48 模拟器重切为两个新 campaign(各 24,
  4 shard × 6 envs)→ 起新 server → 起监督器:
  - `mw-cc-k1cap-v1`:完整 CausalCache + `--max-replacements 1`
    (beam 子集若相对 recent 尾部的替换数 >1 直接拒绝;dedup 别名视为
    tail 内)。拆 k=1 训练 vs 多帧替换部署的分布偏移之雷:受限 k≤1
    仍应大幅击败基线,放开 k 只是系统完备性。
  - `mw-frozensel-v1`:冻结 8B(**无 HGKV adapter**)+ selector 两遍。
    补缺失 baseline:selector 单独是否够?若明显差于完整法,即 HGKV
    必要性的直接证据。
- 存活:同款监督器(断点续跑/心跳/STALL=3),本地 Monitor 7 分钟轮询
  两机心跳,DONE/STALE/ALERT 皆上报;翻臂由本地后台触发器执行并校验
  `H0x_FLIPPED` 哨兵,失败即 ALERT。
- 归约工具入库:`code/scripts/harvest_mobileworld_arm.sh`(双机收割
  task→score,跨机重复任务取首见并报冲突)+
  `code/scripts/reduce_mobileworld_review_ablations.py`(full/62/55
  三列 + 对 CausalCache-B4、HGKV+Recent-4 三轮均值的任务配对
  bootstrap 差)。

## text-only 臂终值(2026-07-29 晨)与一则评测故障

- **mw-cc-textonly-v1 收官 116/117**:`MastodonUpdateContactsTask`(属
  memory-critical 62)在 h01 上策略轨迹正常跑完,但 benchmark 侧
  `/task/eval` 连续 HTTP 500(两个不同模拟器、4 轮监督器尝试、12+ 次内部
  重试),无法产生分数——评测基础设施故障,非臂属性。该任务记 missing,
  配对分析自动按 n=116/61 剔除。h01 由人工点火翻臂(触发器条件是 8/8
  DONE,STALL 卡死了它);h00 触发器已加"7 DONE + 1 STALLED 也翻"容错。
- **结果**(`data/results/mobileworld_review_ablations_v1/`):full 27.6%,
  memory-critical 18.0%,control 38.2%。对完整 CausalCache(像素恢复):
  full −6.6pp(boot p=0.035)、memory-critical **−10.4pp(p=0.006)**、
  control −2.4(ns);对 HGKV+Recent-4:full −2.9(ns)、mem −1.6(ns)。
  即:把恢复槽位换成 verbatim 文本响应,memory-critical 增益整体消失,
  且不优于单纯 Recent-4——增益载体是像素,不是"历史被提及"。

## OCR 臂终值 + h00 自动翻臂(2026-07-29)

- **mw-cc-ocr-v1 收官 117/117**(h00 无 eval 故障):full 32.5%,
  memory-critical 24.2%,control 41.8%。对像素恢复:mem −3.8pp(ns);
  对 Recent-4:mem +4.8pp(ns)。OCR−text-only 直接配对差 +4.3(full)/
  +4.9(mem),CI 跨零。排序:text-only ≈ Recent-4 ≪ 像素(显著),
  OCR 居中(与两端皆不可分)。像素是唯一对 Recent-4 显著胜出的载体。
- h00 触发器 15:45 UTC 点火(8/8 DONE),自动翻臂成功;两机 96 模拟器
  现全部在跑 mw-cc-k1cap-v1 + mw-frozensel-v1(每臂 48,shard 0-7)。

## 32B 跨骨干支线启动(2026-07-29,用户升级为必做)

- **目标**:同 v4 语料/目标在 GUI-Owl-1.5-32B 上重训 HGKV,offline DiD 门控
  证明 token-gated 机制跨骨干泛化(闭环不在此支线范围)。
- **配置**:`causalcache_desktop_did_hgkv_v4_32b.json`(=v4 HGKV,仅换
  32B snapshot manifest;300 步、cadence 50、lr 1e-4、accum 2)。
  world_size=3 → 有效批 6(8B 主线 4×2=8),支线非冻结契约,差异已注记。
- **补丁**:`train_success_sft_lora.py` / `score_sparse_history_arms.py`
  加 `--model-profile {8b,32b}`(与 serve 同款,profile 常量经 serve 侧
  健康检查验证)。
- **运行**:h01 辅容器 `sglang-omni-jaxan-2`,容器内 CUDA 1,2,3 =
  host GPU 5/6/7(host GPU4 被外部 133GB 进程占用,即用户所称"3 卡");
  语料 `/bigdata/mw/desktop-did-corpus-v4`(sha 与 v4 主线同源),fresh
  frozen-score cache(8B 缓存对 32B 无效)。输出
  `/bigdata/mw/runs/desktop-did-32b/hgkv/`。
- **存活**:supervisor 3 次重试 + EXIT_N 哨兵;本地 Monitor 15 分钟轮询
  `global_step`/`step_checkpoint`/Traceback/OOM。教训:该容器 shell 无
  默认 PYTHONPATH,torchrun 必须显式 `PYTHONPATH=code`(首launch三连败)。
- **后续**:训毕用 `score_sparse_history_arms.py --model-profile 32b`
  出 dev 组 DiD 门控报告(borrow 主线口径:DiD select、|A_r|、wrong
  drift、cap 0.02),等 k1cap/frozensel 收官释放 canonical GPU 后跑。

## k1cap + frozensel 终值(2026-07-29):两个反转

| 臂 | full | mem-crit | ctl | vs Recent-4 (mem) | vs CC-B4 (mem) |
|---|---|---|---|---|---|
| CausalCache B4(3 轮均值) | 33.9 | 28.0 | 40.6 | +8.6* | — |
| frozensel(无 HGKV+selector) | 36.8 | 30.6 | 43.6 | +11.3 (p=.006) | +2.7 (ns) |
| k1cap(CC + max-replacements 1) | 28.2 | 19.4 | 38.2 | 0.0 (ns) | −8.6 (p=.02) |

- **反转 1(HGKV 部署非必要)**:frozensel 与完整 CC 统计不可分(数值更高),
  对 Recent-4 显著。selector 训练标签仍由 HGKV 打分产生
  (build_hgkv_coalition_score_cache_v2 强制 --hgkv-checkpoint-sha256),
  即 HGKV 目前的必要性在离线教师端,部署端 adapter 行为中性——与正文
  FR≈HR≈B0 的分解一致,把"增益由 allocation 承载"补完到 selected 侧。
- **反转 2(k≤1 塌回基线)**:硬约束每步最多替换 1 帧后,mem-critical
  恰好回到 Recent-4 水平(19.4 vs 19.4),完整 CC 的 +8.6 全部消失。
  多槽重组(k>1,含 26% 全窗替换)是闭环增益的载体,不是完备性装饰。
  原"k=1 训练接口即可"的叙事需要重写:训练接口 k=1 依旧(离线边际
  证据不变),但**部署端的多槽组合自由是必要条件**。
- 后续(用户指示):用冻结策略重打教师标签训 frozen-teacher selector,
  验证 HGKV 是否连教师端也非必要。打分器已加 --teacher frozen。

## frozen-teacher selector 管线启动(2026-07-29)

- 阶段 1(在跑):singleton 重打分,--teacher frozen(锚+候选全 bypass,
  不注入 adapter)。11 shard:h00 0-6(GPU 0,1,3-7,共 7 卡,
  `/data/runs/selector-v4-frozen/singletons`),h01 7-10(canonical 0-3,
  `/bigdata/mw/runs/selector-v4-frozen/singletons`)。两机 screening
  manifest sha256 一致已核(38889f11…)。每 shard 3 次重试 + jsonl 断点
  续跑;本地 Monitor 15 分钟轮询 heartbeat 汇总 + Traceback 告警。
- 阶段 2:sets 重打分(score_selector_v4_sets --teacher frozen,以 frozen
  singletons 为种子);阶段 3:train_selector_v4_marginal 训 two_tower;
  阶段 4:frozen policy + frozen-teacher selector 上 96 模拟器闭环
  (117 任务,count-8 shard 0-7)。**模拟器舰队为此保留未拆**;双机 8 个
  旧 policy server 已按 PID 杀净(pkill 三层引号失效教训:改脚本文件)。
- 假设检验:若 frozen-teacher selector ≈ HGKV-teacher selector,则 HGKV
  在教师端也非必要,系统故事收敛为"冻结策略 + 预算感知重分配";若明显
  更差,则 HGKV 的必要性 = 教师端效用测量仪,与 frozensel 部署结果自洽。

## 32B 跨骨干 offline DiD 门控(2026-07-29)

- 训练:300 步收官(EXIT_0,world 3 × accum 2,B∈{1,2,4} 混合语料同 v4)。
- 门控(B=1,94 dev 组,与 8B 同口径):5/6 checkpoint 全门通过,按冻结
  规则选 step200:**DiD select +0.0142 [0.0109, 0.0176]**(8B v4 为
  +0.0224 [0.0162, 0.0294]);SA−RA +0.080,SA−WA +0.042,
  A_r ∈ [−0.006, +0.001] 全部远在 0.02 帽内。
- 结论:同配方、同语料、免调参在 4× 骨干上复现"漂移包络内的正选择性"
  ——token-gated HGKV 的跨骨干泛化证据(main_v2 的 pending 可回填)。
