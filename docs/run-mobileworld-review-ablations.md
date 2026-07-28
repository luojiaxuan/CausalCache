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
