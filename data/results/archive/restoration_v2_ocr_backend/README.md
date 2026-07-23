# Restoration-v2 OCR backend evidence

本目录保存 OCR/image backend 的轻量运行证据；ONNX models、真实 screenshots 与完整 derived records 不进入
Git。正式通过前，dependency 5 仍为 pending，且不允许生成 GUI-Owl policy/restoration output。
当前 synthetic、HF model、real-screen 与 HF dataset immutable checks 均已完成，dependency 5 现为 passed；
完整 derived artifact、baselines 与 execution config 仍阻止 policy output。

## Superseded attempts

### `ocr-synthetic-20260715T114552Z-0e1dfa1`

- pushed source：`main@0e1dfa19f1093258dce28f34fdaeab5e3c790dff`，clean detached Hyper00 worktree；
- runtime：Hyper00 `node-radixark-16-0000`，x86_64，Python 3.12.3，container
  `sglang-omni-jaxan-07151357` / image
  `sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`，CPU only；
- UTC bracket：`2026-07-15T11:45:52.110804304Z` -- `2026-07-15T11:45:53.639900155Z`；
- full argv：

```text
/data/.venv/causalcache-ocr-v2/bin/python -m scripts.validate_restoration_v2_ocr_backend inspect-golden --backend-config configs/restoration_v2_ocr_backend.json --model-dir /data/artifacts/causalcache-ocr-ppocrv5-mobile-v1 --wheel-dir /data/tmp/causalcache-ocr-v2-wheels --fixture ../data/fixtures/restoration_v2_ocr_golden.json --output /data/tmp/causalcache-ocr-v2-golden-0e1dfa1-repeat-1/inspection.json
```

- observed forward：synthetic text 得到 `Causal Cache / Step 42`；model、wheel、installed source 与
  436-entry embedded character identity 均通过；
- invalidation：`rapidocr_package_file_sha256` 曾用 basename 作为 key，导致
  `inference_engine/onnxruntime/main.py` 与 `ch_ppocr_rec/main.py` 冲突，后者覆盖前者。虽然两个 source
  file 在 runner 内都验过 SHA，这份 serialized evidence 无法同时证明二者，故标为
  `INVALID_EVIDENCE_SCHEMA_PACKAGE_PATH_COLLISION`；
- disposition：未回写 fixture expected，不作为 golden repeat，也不上传 HF。该 attempt 未使用 GPU、未加载
  GUI-Owl，未产生 policy/restoration output。

后续 runner 必须用 RapidOCR package-relative path 作为 source evidence key，并从新的 pushed commit 重新执行
两个独立进程。

## Frozen expected candidate

`main@09e4f6d2c1c91bc8e17cb19c4da8c22153919d45` 修复 package-relative key 后，在同一 Hyper00
CPU runtime 运行两个独立 `inspect-golden` process：

| Repeat | UTC bracket | Canonical inspection SHA256 |
| --- | --- | --- |
| 1 | `2026-07-15T11:47:47.140669493Z` -- `2026-07-15T11:47:48.656088201Z` | `6cda74bb4f33708795842c947dac53e380e9f0d4e3debcbf9bb38334a645af44` |
| 2 | `2026-07-15T11:47:57.685531727Z` -- `2026-07-15T11:47:59.202436027Z` | `6cda74bb4f33708795842c947dac53e380e9f0d4e3debcbf9bb38334a645af44` |

两次 full argv 与 superseded attempt 相同，仅 output path 分别为
`/data/tmp/causalcache-ocr-v2-golden-09e4f6d-repeat-{1,2}/inspection.json`；两次均从 clean detached
`09e4f6d` 执行。canonical records byte-identical，synthetic OCR tokens 是
`["Causal", "Cache", "Step", "42"]`，record SHA256 为
`47941b52bf42df7c452119821e824953ab1a3ebd2aa8b1d31ef97b21d1093371`。

这些 expected fields 已写入 Git fixture。fixture 的 `status` 永久描述内容为
`synthetic_expected_inspection_frozen`；是否通过独立 validation 只由本 result record 描述，避免运行后改
status 又改变 fixture SHA。

## Superseded validation of lifecycle-labeled fixture

从 pushed `main@b82c3d8a4a672af4c1802f8772861e235926c39c` 两次运行 `validate-golden` 均返回
`PASSED_OCR_GOLDEN_VALIDATION`，UTC brackets 分别为
`2026-07-15T11:51:36.090924406Z--11:51:37.643951115Z` 与
`2026-07-15T11:51:51.246038518Z--11:51:52.779843042Z`；两份 validation JSON byte-identical，SHA256
均为 `e1fee0870d8c644e1aedc2c064730155f7273d6e4c4bda511b1bcedcb2e65245`，fixture SHA256 为
`b407ae14ba2d06e63432b58f70ab10a1b070425aa5d6934e259fc7cff6ab2015`。

该 fixture 的 expected 内容正确，但顶层 status 当时仍包含 mutable `validation...pending` 字样。为避免
通过后改 status 破坏 hash-bound fixture，status 被改成永久内容描述；因此上述 passing validation 作为
superseded lifecycle attempt 保留，最终 synthetic verdict 必须从包含静态 status 的新 pushed commit 再跑。

## Final synthetic verdict

从 pushed `main@820fa54c5e65e72e0f3395ea606f1b5dc71bf71f` 的 clean detached worktree 两次运行
`validate-golden`，两次均返回 `PASSED_OCR_GOLDEN_VALIDATION`，validation JSON byte-identical，SHA256
均为 `3f4fde7c58b31813af630d6646ac5ef353856ca6eff7add7aa31ddcb7a3c49a6`。最终 static fixture SHA256 为
`8c81feb37af9d2393d62d6ddbfe65948f84e0eb0be0c0baf10126c547103bca6`。

完整 argv、UTC bracket、container identity、model/wheel hashes 与 negative declarations 位于
[`synthetic_summary.json`](synthetic_summary.json)。synthetic golden 至此 passed；当时 private HF model
immutable revision、6-image policy-blind real-screen golden 与 source/artifact manifest 均 pending。后续
model artifact 里程碑见下节。

## Immutable HF model artifact

三份 ONNX、model card 与 artifact manifest 已上传到 private HF model
`gavinlaw/causalcache-rapidocr-ppocrv5-mobile-en@v1.0.0`；tag 解析到 full revision
`0dbc766a73ee88d10d52285d434dbfec58617835`。`2026-07-15T12:01:12Z` 从该 revision fresh
re-download 后 6/6 file sizes/SHA256 一致。该里程碑当时绑定 8 个 backend source hashes；完成态 manifest
在 real-screen 通过后扩展为 14 个 source，见
[`restoration_v2_ocr_backend.json`](../../../manifests/restoration_v2_ocr_backend.json)。离线
`artifact-source` validator 输出 `PASSED_OCR_ARTIFACT_SOURCE_VALIDATION`。

因此 model artifact 与 synthetic golden 已通过；当时仍待 6-image policy-blind real-screen golden。后续
完成态见下节。

## Final real-screen verdict

从 pushed `main@dcc6e217b4885cef5f745d987a1ec74e57109717` 在 Hyper00 CPU runtime 两次独立
materialize，并分别从 raw Parquet 重建候选池、重放 6 次 OCR。两个 5-file trees byte-identical，SHA256
均为 `605d6396b0cde84697ff3f2407a3630d7ccdb822f55c2bdae56be82e942a7e25`；两次 validator 均返回
`PASSED_REAL_SCREEN_ARTIFACT_VALIDATION`。

payload 已上传 private HF dataset
`gavinlaw/causalcache-guiodyssey-restoration-v2-mobile@ocr-real-screen-golden-v1.0.0`，immutable revision 为
`9ebbbbbc4666e8a065f4ecb5240491c70f05e21b`。从全新 cache 按该 revision 下载后 5/5 files 与上传前
size/SHA 一致；fresh snapshot 在 Hyper00 第三次通过同一 raw-source + OCR replay validator。完整 argv、
UTC brackets、runtime、5-file inventory、HF operation metadata 与 negative declarations 见
[`real_screen_summary.json`](real_screen_summary.json)。完成态 manifest 的离线 validator 现返回
`dependency_5_closed=true`。

本步骤未使用 confirm images、未加载 GUI-Owl、未生成 policy/restoration output。OCR dependency 5 已闭合，
但不能替代完整 derived artifact、baseline source hashes 或 execution config。
