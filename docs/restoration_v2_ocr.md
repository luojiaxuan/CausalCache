# Restoration-v2 OCR 与 image backend

## 当前状态

OCR/image implementation identity 已在任何 v2 policy output 前冻结。当前仅关闭了 source/config/interface
层：Hyper00 的 locked CPU runtime 与三份 ONNX 文件已 staged 并验证 SHA；private HF model revision、
end-to-end OCR golden 和 source-hash manifest 仍 pending。因此八项 pre-output dependencies 中第 5 项尚未
标为 passed。

hash-bound backend config 只保存永久静态 identity、预定 private HF repo 与 future artifact-manifest path；
upload status 和 immutable revision 不写进该 config，避免上传后改变 config SHA。当前 live status 由本页、
顶层 README 与未来 `data/manifests/restoration_v2_ocr_backend.json` 共同索引。

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
resize case 与两行 English OCR case。当前 image bytes/prepared hashes 已冻结，OCR expected output 等待从已
push commit 在 Hyper00 两个独立进程中生成后回写。

real-screen golden 不允许人工挑图。eligible pool 只来自 frozen `v2_label_train`/`v2_development` states 的
`candidate_post_state` 与 `current_observation`，先按 image SHA 去重，再分别在 portrait/landscape 中按
`(image_sha256, image_member_path)` 取前 3 张，共 6 张；任一 stratum 不足 3 张即 fail closed。confirm role
完全排除。该规则已写入 backend config，必须先于任何 OCR/policy result 执行。

配置层验证不要求本机安装 OCR extra：

```bash
make validate-restoration-v2-ocr-config
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

正式完成条件：三模型上传 private HF model repo 并 immutable re-download 验证；synthetic golden 两进程
byte-identical；label/development policy-blind real-screen golden 上传 derived HF dataset；Git manifest 绑定
backend/config/module/validator/fixture/lock 与 HF revisions。confirm images 不参与 golden selection。
