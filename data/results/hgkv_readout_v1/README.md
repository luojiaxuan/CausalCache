# HGKV-readout singleton features v1

## 状态

`DONE`。Hyper01 持久化 run root：
`/data02/jaxan/runs/hgkv-readout-v1/`。

- rows：75,628；
- unique `(pair_group, singleton_event_step_id)`：75,628；
- feature dimension：1,280；
- 验证：八个 shard 全部 JSON 可解析、key 无重复、维度一致、逐行 feature 长度正确、
  所有 feature 值有限；
- 总大小：1,122,716,058 bytes；
- artifact 状态：`PENDING_HF_UPLOAD`。

这是 selector 训练输入，不是 benchmark 科学结果。

## Provenance

| 项目 | 值 |
|---|---|
| source commit | `d6973a78cb3fb8fab19795b62a6ebeedcb197eca` |
| singleton `samples.jsonl` SHA256 | `68fef3bb6848f7c949640efa7f51123c14c783625862dfb7bf8f41a6dab2eaa0` |
| hg-s100 checkpoint SHA256 | `8f2cc49e1aa0b06ce231eb54937d813317f5274a799c97b09be7fdb22be46317` |
| model | `GUI-Owl-1.5-8B-Instruct`，仓库 snapshot contract |
| adapter | History-Gated KV，rank 8，alpha 16，最后 8 层 |
| host / GPU | hyper01 / NVIDIA H200，恢复映射 GPU 0/1/2 → shard 1/5/6 |
| completion time | `2026-07-24T08:43:56.014628+00:00` |
| runtime evidence | run root `DONE`、`launch-manifest.json`、`recovery/causalcache-hgkv-recovery-{preflight,plan}.json`、`recovery/causalcache-hgkv-recovery-launch-gpu-snapshot.csv` |

## Shard manifest

| shard | rows | bytes | SHA256 |
|---:|---:|---:|---|
| 0 | 9,454 | 140,347,885 | `8053fa565109e6585fb5ede07556a1c64273e5b2cd6e80d782600f5f61e160ee` |
| 1 | 9,454 | 140,343,727 | `57957c8eba696f502d900d6e4d23a63536ffe5e24f54c1b288f6a68524cc2999` |
| 2 | 9,454 | 140,347,209 | `4142cf697b087d28481acfe840e7d284a7bd1b1b68fb15d2a83ca67f0932fd62` |
| 3 | 9,454 | 140,338,358 | `c5aa58c8dab656ffb91c48d42dafc9a6c2c3cb9101209ad1bb6304c16cb73925` |
| 4 | 9,453 | 140,333,050 | `f381eb9c51f76b1a7c3f547582c6679cc1ab36bb0ff255719651cb160f137e46` |
| 5 | 9,453 | 140,335,372 | `cde9bee8e6f8c7d180dbd14f0dc2271e2bd76a2b178b2b5d46c6b2c378ca4733` |
| 6 | 9,453 | 140,335,301 | `76b466ec32c0654494f14928379fdeb470c6c625957f3532c1c2b17336ca049c` |
| 7 | 9,453 | 140,335,156 | `f12db70256dccd081359633eb7fbc20e4f1189bfa4de58c32e14e5a8144fdd8b` |

## 截断图片恢复

首轮 shard 1/5/6 在同一张
`images/9471050986960951/observation-004.png` 处失败，停在
7,372/7,371/7,372 行。坏文件为 8,704 bytes，SHA256
`9e965ba01e38d923a3800c45683ff1722e2a147c279cbaae07a482d7982f9dca`；
PNG IDAT chunk 声明 8,192 bytes 时只剩 430 bytes。

对 `samples.jsonl` 引用的 14,680 张唯一 PNG 做逐 chunk 长度与 CRC 扫描，只发现该
一张坏图。两个独立持久化副本均为 231,296 bytes，SHA256
`c6d222fca77bb3827c3dc6b1e19e5a1355b012cbc4d71dc81fdfa25bc8e6a70d`；
用一致副本原子替换后重扫为 `bad_count=0`。旧坏文件保存在：

`/data02/jaxan/runs/hgkv-readout-v1/recovery/9471050986960951-observation-004.png.truncated`

没有启用 Pillow 的 truncated-image 容错，因此恢复没有静默解码不完整像素。

## Sparse resume 命令

```bash
env PYTHONPATH=code python3 -m scripts.run_hgkv_readout_extraction_shards \
  --repository-root /data/worktrees/causalcache-d6973a7 \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --dataset-root /data/artifacts/sft/ody-labels-single \
  --output-root /data/runs/hgkv-readout-v1 \
  --lora-checkpoint /data/runs/hgkv-eval/hg-s100.pt \
  --lora-rank 8 \
  --lora-alpha 16 \
  --adapter-layer-count 8 \
  --shard-count 8 \
  --shard-indices 1,5,6 \
  --gpu-local-indices 0,1,2 \
  --expected-rows 75628 \
  --expected-dataset-sha256 68fef3bb6848f7c949640efa7f51123c14c783625862dfb7bf8f41a6dab2eaa0 \
  --expected-checkpoint-sha256 8f2cc49e1aa0b06ce231eb54937d813317f5274a799c97b09be7fdb22be46317 \
  --source-commit d6973a78cb3fb8fab19795b62a6ebeedcb197eca
```
