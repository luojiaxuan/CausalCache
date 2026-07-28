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
