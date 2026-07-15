# Restoration v2 完整派生物完成证据

本目录记录 `causalcache_restoration_v2` dependency 1 的正式完成证据。机器可验证事实位于
[`summary.json`](summary.json)，入口 manifest 位于
[`../../manifests/restoration_v2_derived_artifact.json`](../../manifests/restoration_v2_derived_artifact.json)。
本次工作严格停在 policy-blind derived-data 阶段，没有加载 GUI policy、读取 confirm policy score，
也没有生成 policy 或 restoration output。

## 完成结论

- frozen builder：Git
  `1a01f2323647d092cab67f0531ecb877a4a255de`，tree
  `753e42ab9663553d461e8200dcfcdd5576ee209e`；正式 Hyper00 checkout 在执行时满足
  `HEAD == origin/main` 且 clean。Hyper00 无法用 non-interactive HTTPS credential fetch，因而使用从 Mac
  已验证 pushed `main` 生成的 complete Git bundle；bundle 为 1,325,083 bytes，SHA256
  `a97563e9ddbbc2ab2242af6070480aae518938fd2100d4c09b808b7b6c5e1857`。
- 两次独立 materialization + builder-integrated 210-record OCR replay 均通过，UTC 分别为
  `2026-07-15T14:26:37.779182367Z--14:47:57.486955852Z` 与
  `2026-07-15T14:51:24.170577043Z--15:12:43.238635504Z`。
- 两次均得到 35 trajectories、175 events、65 states、210 images、210 OCR records；exact six files
  byte-identical。artifact tree SHA256 为
  `475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e`，OCR aggregate 为
  `1e04ddbdd2fd6e5fc50206c436f512908b269fc95566476c799a075eb9af7010`。
- private Hugging Face dataset
  `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile@restoration-v2-derived-v1.0.0`
  通过 `HfApi.dataset_info(revision=tag).sha` 解引用到 immutable revision
  `89f136abaff797e14fe758a198996e51032a10a6`。
- immutable exact-six clean projection 于 Hyper00 在
  `2026-07-15T15:20:56.587011300Z--15:31:30.534390269Z` 完成第三次 standalone 210-record replay；
  tree、OCR aggregate 和逐文件 bytes 均与两次 materialization 一致。

## Exact six-file identity

| Path | Bytes | SHA256 |
| --- | ---: | --- |
| `.gitattributes` | 42 | `116c15b65825b94cdce725a6c380e58928b054dc861b6b7b07f5eae549a7c2e9` |
| `README.md` | 2,107 | `0635bba67aad567fc75427c55bd8b1778b92e60499a6c1fecb4cfd90390c0918` |
| `derived/restoration-v2-v1/images-00000-of-00001.tar` | 155,883,520 | `1a383c116117d26d7a123343816187695c8312a5923359a6518d7f0051d17326` |
| `derived/restoration-v2-v1/manifest.json` | 7,382 | `90a48f794123b915994df9a780a0553f71f9fad26c95cae7b24d1435d1627945` |
| `derived/restoration-v2-v1/ocr-records-00000-of-00001.jsonl` | 1,897,573 | `b8c96b9951d8166839dde71b4c8ca60f099b92e892bbeafde819beb96a9fc97d` |
| `derived/restoration-v2-v1/trajectories-00000-of-00001.jsonl` | 595,807 | `c08ae40d0fe27783c03c00459066db43dd8610cc8b77f74babdcc97dc18b747e` |

## HF 发布、重下载与既有 artifact 保留

上传使用 Mac staging
`/tmp/causalcache-restoration-v2-derived-1a01f23-repeat-1`；它与 Hyper00 原始 repeat-1 路径
`/data/tmp/causalcache-restoration-v2-derived-1a01f23-repeat-1` 分开记录。两者的 exact-six file/tree identity
一致，因此跨机器 transport 没有改变 bytes。上传只包含 exact six allowlist，没有使用 `--delete`。
summary 中的 build、upload、tag、download 与 replay 完整 argv 来自执行 transcript；builder/validator JSON
日志和 HF CLI JSON stdout 本身没有内嵌 argv，不能把这些结果文件误称为 argv 的独立来源。Mac upload 原始
时间文件因 BSD `date` 不支持 `%N`，分别写成了字面量 `2026-07-15T15:15:31.NZ` 与
`2026-07-15T15:16:51.NZ`；summary 只把其中可观察、无歧义的秒级字段规范化为
`2026-07-15T15:15:31Z` 与 `2026-07-15T15:16:51Z`，没有伪造纳秒精度。

第一次 download preflight 于 `2026-07-15T15:17:45Z` 启动，但命令同时指定 `--cache-dir` 与
`--local-dir`，被 `hf` CLI 的互斥参数检查在网络下载前拒绝。该 attempt 的 JSON stdout 为 0 bytes，目标
`/tmp/causalcache-restoration-v2-hf-download` 未创建，因而没有任何 dataset file 落盘；它是已废弃的 preflight
failure，不计入正式 redownload evidence。

正式 attempt 2 使用新的空目录
`/tmp/causalcache-restoration-v2-hf-download-attempt-2-89f136a`，于
`2026-07-15T15:18:07Z--15:18:16Z` 按 immutable revision 下载成功。`--local-dir` 自动生成的
`.cache/huggingface` metadata 没有进入验证输入；Mac 随后建立只含 exact six files 的 clean projection，
再按逐文件 hash 搬运到 Hyper00
`/data/tmp/causalcache-restoration-v2-derived-1a01f23-hf-redownload` 供 standalone validator 使用。

上传后 `main@89f136abaff797e14fe758a198996e51032a10a6` 恰有 9 files：本次 exact six projection，外加原有
`golden/real-screen-v1/{images-00000-of-00001.tar,manifest.json,ocr-records-00000-of-00001.jsonl}` 三文件。
旧 tag `ocr-real-screen-golden-v1.0.0` 仍通过 `dataset_info(tag).sha` 解引用到
`9ebbbbbc4666e8a065f4ecb5240491c70f05e21b`，其五文件 snapshot 保持可复现；HF annotated-tag object SHA
与解引用后的 dataset revision 不应混用。

## Runtime 与边界

三次 OCR replay 均在 Hyper00 container `sglang-omni-jaxan-07151357` 的
`CPUExecutionProvider` 上执行，使用 pinned RapidOCR 3.8.4、ONNX Runtime 1.24.4、Pillow 12.2.0 与
PyArrow 24.0.0；没有使用 GPU，dtype 为 `not_applicable_cpu_onnxruntime`，无随机 seed。builder 直接从
pinned raw GUIOdyssey source 和 pinned OCR runtime 生成 records，没有消费外部 OCR JSONL。

本结论只闭合 pre-output dependency 1：证明完整 derived artifact 已两次确定性构建、发布为 immutable HF
revision，并在 fresh redownload clean projection 上完成第三次 exact replay。它不证明任何 restoration
attribution 或 policy 效果；execution config 仍须独立冻结后才能打开 policy inference。

轻量完成证据可在任意保留历史 Git object 的 checkout 中直接复验：

```bash
cd code
python3 -m scripts.validate_restoration_v2_derived_artifact \
  --artifact-manifest ../data/manifests/restoration_v2_derived_artifact.json \
  --repository-root ..
```

本 milestone 不新增根 `Makefile` target，因为历史 builder 已绑定当时 `Makefile` 的 SHA；直接 CLI 可避免
完成态提交反向改变已发布 artifact 的 frozen source identity。
