# Restoration-v2 OCR 与 image backend

## 当前状态

OCR/image implementation identity 已在任何 v2 policy output 前冻结。Hyper00 locked CPU runtime、synthetic
golden 与 private HF model artifact 均已验证；`v1.0.0` 解析到 immutable revision
`0dbc766a73ee88d10d52285d434dbfec58617835`，6/6 files 已 fresh re-download 并逐文件验 hash，Git
source/artifact manifest 也已通过 fail-closed validator。6-image real-screen golden 与完成态 manifest
现已通过，因此八项 pre-output dependencies 中第 5 项已标为 passed。real-screen 的 17-file pre-output
source contract、materializer 与独立 artifact validator 已冻结；source contract SHA256 为
`374a38c997a1ee9a715a8cf6ce9b7ca26edc1cf56f503c2d42a97436afac16c5`。Hyper00 两次独立 materialization、
两次独立 replay 与 HF immutable re-download 后第三次 replay 全部通过；private dataset revision 为
`9ebbbbbc4666e8a065f4ecb5240491c70f05e21b`。

hash-bound backend config 只保存永久静态 identity、预定 private HF repo 与稳定 artifact-manifest path；
upload status 和 immutable revision 不写进该 config，避免上传后改变 config SHA。当前 live status 由本页、
顶层 README 与
[`data/manifests/restoration_v2_ocr_backend.json`](../data/manifests/restoration_v2_ocr_backend.json) 共同索引。

## 选择

GUIOdyssey 没有 accessibility tree，所以 offline derived artifact 固定使用：

- `RapidAI/RapidOCR v3.8.4@86b48d2c96818f4dd7897627b04cdaf2c1dc1172`；
- `onnxruntime==1.24.4`，仅 `CPUExecutionProvider`；
- PP-OCRv5 mobile detector `ch_PP-OCRv5_det_mobile.onnx`；
- PP-OCRv5 mobile English recognizer `en_PP-OCRv5_rec_mobile.onnx`；
- orientation classifier 关闭；intra/inter-op threads 都是 1。

选择 unified `rapidocr` 而不是逐渐停止维护的 `rapidocr_onnxruntime`，与
[官方安装说明](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/install/) 一致。版本固定在
3.8.4，并显式选择 PP-OCRv5，不依赖后续版本的默认模型。35 条 v2 trajectory 的 task instruction 全为
English；recognizer 语言是在读取 policy output 前按数据属性确定，不是按 OCR 结果选择。

完整参数位于
[`restoration_v2_ocr_backend.json`](../code/configs/restoration_v2_ocr_backend.json)，runtime lock 位于
[`restoration_v2_ocr_lock.txt`](../code/requirements/restoration_v2_ocr_lock.txt)。关键 transport identity：

| 组件 | SHA256 |
| --- | --- |
| `rapidocr-3.8.4-py3-none-any.whl` | `1a8400df99ea2348a6d1902d1c6fa64c29edcf56b3c58cecd4cf1e5ff51b7ae2` |
| `onnxruntime-1.24.4` CPython 3.12 x86_64 wheel | `0d640eb9f3782689b55cfa715094474cd5662f2f137be6a6f847a594b6e9705c` |
| PP-OCRv5 mobile detector | `4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae` |
| PP-OCRv5 English mobile recognizer | `c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8` |
| inactive PP-OCRv4 classifier | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` |

RapidOCR 3.8.4 即使 `Global.use_cls=false`，constructor 仍创建 classifier session。因此第三个模型虽然不参与
OCR forward，也必须显式传入、hash-pinned 并放进 model artifact，不能把它当成不存在的 hidden
dependency。English recognizer 的 436-entry character inventory 内嵌在 ONNX `character` metadata，UTF-8
SHA256 为 `e025a66d...70d6`；validator 同时检查 installed `config.yaml`、`default_models.yaml` 与该 inventory，
不能只相信 package version 字符串。

## Image preprocessing

OCR engine 直接读取 original encoded PNG bytes，不读取下面的 256x256 resize。RapidOCR bytes branch 固定
经过 `PIL.Image.open(BytesIO(raw))`、NumPy RGBA、float32 alpha composite、`cv2.COLOR_RGB2BGR`；selected
screenshots 必须是 opaque RGBA，因此 background choice 不影响 pixel，但 `load_image.py` 仍以 exact source
SHA 绑定。原始 bytes 另由 Pillow 12.2.0 为 MAD/RGB baseline 执行：

```text
Image.open(BytesIO(raw)).load()
image.convert("RGB")
image.resize((256, 256), resample=Image.Resampling.BILINEAR)
```

不执行 `exif_transpose`，不传 `box` 或 `reducing_gap`。screen-change MAD 和 RGB baseline 都消费该 backend
产生的 row-major uint8 RGB bytes。当前选中 35 条 trajectory 的 353 个 unique screenshots 预检查均为
opaque RGBA PNG、无 EXIF；derived builder 会重新逐图 fail closed 检查，而不是信任这条统计。

## OCR record

每张图保存 uncapped screen-level OCR，而不是只保存 summary delta：

```text
image_member_path / image_sha256
source format/mode/width/height/EXIF/alpha identity
raw RGB SHA / resized 256x256 RGB SHA
backend config SHA
nodes:
  polygon_xy
  bbox_top_left_bottom_right
  raw_text / normalized_text
  confidence_decimal_string
full_spatial_tokens / full_spatial_tokens_sha256
canonical_ocr_record_sha256
```

polygon 坐标按 nonnegative `floor(x+0.5)` 转为原图 pixel 并 clip；node 先按
`top,left,bottom,right,normalized_text` 排序，再用 raw text、confidence 与 polygon 做 deterministic
tie-break。text 使用 NFKC、collapse whitespace、strip；full tokens 不截断，供 OCR-token Jaccard baseline
使用。只有写入 strong low-fidelity summary 的 added/removed delta 每侧截断到 32，并记录 discarded count。

## Golden 与执行

Git fixture
[`restoration_v2_ocr_golden.json`](../data/fixtures/restoration_v2_ocr_golden.json) 内嵌两个固定 PNG：2x2 RGB
resize case 与两行 English OCR case。image bytes/prepared hashes 以及 pushed `main@09e4f6d` 两个独立
进程产生的 byte-identical OCR expected fields 已冻结；随后从最终 pushed `main@820fa54` 两个独立进程运行
`validate-golden`，均得到 `PASSED_OCR_GOLDEN_VALIDATION` 且 validation JSON byte-identical。完整 evidence
见 [`data/results/restoration_v2_ocr_backend/`](../data/results/restoration_v2_ocr_backend/)。

real-screen golden 不允许人工挑图。eligible pool 只来自 frozen `v2_label_train`/`v2_development` states 的
`candidate_post_state` 与 `current_observation`，先按 image SHA 去重，再分别在 portrait/landscape 中按
`(image_sha256, image_member_path)` 取前 3 张，共 6 张；任一 stratum 不足 3 张即 fail closed。confirm role
完全排除。该规则已写入 backend config，必须先于任何 OCR/policy result 执行。

policy-blind raw-source 复算得到 45 states、180 occurrences（135 candidate post-state + 45 current）、75
unique images，其中 55 portrait、20 landscape、0 square。confirm role 有 97 张 unique images，与 eligible
pool 的 SHA intersection 为 0。同 SHA 的全部 occurrences/path 都保留并逐路径验 bytes，canonical
representative 固定为最小 member path。冻结的 6 张图为：

| Stratum | SHA256 | GUIOdyssey member | Shape |
| --- | --- | --- | --- |
| portrait | `02a92f2d9749446e03533884f0dccce0fadff860a07e3192414fe60f6af63c3f` | `images/0131649930078879/observation-005.png` | 720x1280 |
| portrait | `0e3304250dcf14ea3fb2c19cc4e14a2716b7cc77d00966c3cbc6d9e1cde4031a` | `images/0217738978329323/observation-004.png` | 1440x3120 |
| portrait | `112d550f8bc65da5541015d7ff68677c410865f57a629144e9214ba6bb1e3cd1` | `images/0217738978329323/observation-003.png` | 1440x3120 |
| landscape | `11cbacfa5532c1839970ea53db979df9bcc6a12ab91839ac284ca7af903b054b` | `images/0214300008821039/observation-003.png` | 2560x1600 |
| landscape | `1c68bb93daceb610845ba368ba3015e7c3a86336c9271f062c6b9e61d7a7ef5c` | `images/0119685762769531/observation-003.png` | 2560x1600 |
| landscape | `250c54400ebcffa05d2085c6ba2127d226732aea577792fa2dc96bf0b4d38127` | `images/0214993880872733/observation-004.png` | 2560x1600 |

source contract 是
[`restoration_v2_real_screen_source.json`](../data/manifests/restoration_v2_real_screen_source.json)。正式
payload 固定为 `.gitattributes`、`README.md`、`golden/real-screen-v1/images-00000-of-00001.tar`、
`ocr-records-00000-of-00001.jsonl` 与 `manifest.json`。tar 只含上述 6 张原始 PNG bytes，并使用
mtime/uid/gid 为 0 的 deterministic USTAR；validator 必须从 raw Parquet 重建候选池、重放 OCR，逐字节
比较 USTAR、JSONL 和 manifest，再比较完整 5-file artifact-tree hash。该 prefix 只闭合 OCR behavioral
golden，不替代完整 derived dataset dependency。

### Formal real-screen verdict

formal source 是 pushed `main@dcc6e217b4885cef5f745d987a1ec74e57109717`。Hyper00 CPU-only 两次
materialization 的 UTC brackets 为 `12:56:51.799980862Z--12:57:21.514769385Z` 与
`12:58:23.052897194Z--12:58:53.066616371Z`；两个独立目录的 5 个文件全部 byte-identical。对应 raw-source +
OCR replay 分别在 `12:57:39.541455453Z--12:58:09.121744442Z` 与
`12:59:06.804010548Z--12:59:36.733867218Z` 通过。canonical identities：

| Item | SHA256 |
| --- | --- |
| complete 5-file tree | `605d6396b0cde84697ff3f2407a3630d7ccdb822f55c2bdae56be82e942a7e25` |
| payload index | `5d3e061b7db176ff6d6e97939cb98c2a6b819a928cea78d35677cc54cd0b1eb5` |
| raw-image USTAR | `9876d1e604634712374c3b53cc1e7ef40141daf415b6699941ccc55dbcc759f7` |
| OCR JSONL | `c646f1a182f1a6b6c43a17115f15c799e82ca283dc94844c01b8dd77b1ff2a70` |
| payload manifest | `c9652993d8a9854f0a35de596176b9c83ff127240ee21548180e20bd4df88cab` |

exact tree 已上传 private HF dataset
`gavinlaw/causalcache-guiodyssey-restoration-v2-mobile@ocr-real-screen-golden-v1.0.0`，tag 解析到 full immutable
revision `9ebbbbbc4666e8a065f4ecb5240491c70f05e21b`。从全新本机 cache 按该 revision 下载后 5/5 file
size/SHA 一致，再送回 Hyper00 于 `13:02:39.070364572Z--13:03:08.754529856Z` 完成第三次 raw-source + OCR
replay。完整 argv、runtime、HF upload/download 和 negative declarations 见
[`real_screen_summary.json`](../data/results/restoration_v2_ocr_backend/real_screen_summary.json)。confirm image、
policy output 与 restoration output 均未使用或生成。

canonical model artifact 是 private HF model
`gavinlaw/causalcache-rapidocr-ppocrv5-mobile-en@v1.0.0`；full immutable revision 为
`0dbc766a73ee88d10d52285d434dbfec58617835`。model card、artifact manifest、三份 ONNX 与
`.gitattributes` 共 6 个文件已在 `2026-07-15T12:01:12Z` 用 HF CLI 1.23.0 fresh re-download，6/6
size/SHA256 一致。Hyper00 `/data/artifacts/causalcache-ocr-ppocrv5-mobile-v1` 现在仅是可重建 cache。

配置层验证不要求本机安装 OCR extra：

```bash
make validate-restoration-v2-ocr-config
make validate-restoration-v2-ocr-artifact
```

Hyper00 runtime 使用 Python 3.12.3 持久 venv，不使用 GPU。下面命令从 exact lock 重建 runtime，
`--no-deps` 防止 editable project install 再解依赖；正式 golden 还会逐字节验证 staged RapidOCR/ONNX Runtime
wheel 与三份 model 文件：

```bash
python3 -m venv /data/.venv/causalcache-ocr-v2
/data/.venv/causalcache-ocr-v2/bin/python -m pip install \
  -r /data/repo/code/requirements/restoration_v2_ocr_lock.txt
/data/.venv/causalcache-ocr-v2/bin/python -m pip install --no-deps -e /data/repo
/data/.venv/causalcache-ocr-v2/bin/python -m pip check
cd /data/repo/code
```

正式运行命令：

```bash
/data/.venv/causalcache-ocr-v2/bin/python -m scripts.validate_restoration_v2_ocr_backend \
  inspect-golden \
  --backend-config configs/restoration_v2_ocr_backend.json \
  --model-dir /data/artifacts/causalcache-ocr-ppocrv5-mobile-v1 \
  --wheel-dir /data/tmp/causalcache-ocr-v2-wheels \
  --fixture ../data/fixtures/restoration_v2_ocr_golden.json \
  --output /data/tmp/causalcache-ocr-v2-golden/inspection.json
```

private HF model immutable re-download、synthetic golden 两进程 byte-identical、label/development
policy-blind real-screen golden、private HF dataset immutable re-download 与完成态 Git manifest 已全部完成。
OCR dependency 5 现在 passed。该结论只解锁一个 pre-output dependency；完整 derived dataset、baselines 与
execution config 仍 pending，不能据此启动 policy output。confirm images 未参与 golden selection。
