# Restoration-v2 OCR backend evidence

本目录保存 OCR/image backend 的轻量运行证据；ONNX models、真实 screenshots 与完整 derived records 不进入
Git。正式通过前，dependency 5 仍为 pending，且不允许生成 GUI-Owl policy/restoration output。

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
