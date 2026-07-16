# Restoration v2.2-eager full-45 substrate

本目录只保存 Git 轻量结果。正式 raw evidence 的 canonical source of truth 是 private Hugging Face dataset：

- repo：`gavinlaw/causalcache-restoration-v2-2-eager-full-45-substrate-mobile`；
- tag：`v2.2-eager-full-45-substrate-v1`；
- immutable revision：`3577099d505b8c652d764f41269df911128ec767`；
- path：`raw/restoration-v2-2-eager-full-45-substrate-v1.tar`。

## 结论

唯一 fresh-45 attempt 的正式结论是：

`PASS_V2_2_EAGER_FULL_45_SUBSTRATE`

| 指标 | 结果 |
| --- | ---: |
| completed / attempted states | 45 / 45 |
| parse coverage | 45 / 45 |
| exact repeat canonical-action agreement | 45 / 45 |
| finite-logit coverage | 45 / 45 |
| finite repeat KL | 45 / 45，mean = 0.0 |
| memory-sensitive states | 45 |
| generation / teacher / KL | 90 / 135 / 90 |
| retry / top-up | 0 / 0 |
| restoration / gate-training / confirm | 0 / 0 / 0 |

因此 eager fixed-seed/TF32-off substrate gate 已通过，可以另行冻结后续 restoration attribution source；本结果
本身不包含 restoration label、memory selector、gate checkpoint、confirm-policy output 或 closed-loop success。

## 运行与绑定

- execution source：`main@8ae07519f14ac3635f292ee93a7b6d624507427e`；
- formal implementation freeze：`b3a6303d69b1145fbf195e0bd18b9b3065a6f213`；
- contract SHA256：`f473bb8a1657072235dd73bf78a93aff27b438d7baca65b7ef6096cf985effa7`；
- formal source inventory：50 files，SHA256
  `bb6351e4dd5b4470ed1add86dbcbaa714a56063ba8ecf07268b6f13f630f9e54`；
- run-contract SHA256：`397f8da2b89b495c93ef3a034dfe16fb6d883ab4fdb7b830003397c9cc9a2727`；
- Hyper00 / container
  `251556c671584cfb09bcddd7223093bb7f7efd058b779289a90ff515d9e34420` / exact image digest
  `sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
- worker topology：logical `cuda:0` even 23 states、logical `cuda:1` odd 22 states；
- execution window：`2026-07-16T06:05:09.464587Z`--`2026-07-16T06:15:19.889274Z`
  （610.425 秒）；
- steady-state monitor windows 为 92%--100% average utilization；weight-load warmup 前无 GPU process，尾部只剩
  even worker 时单卡 100%，均按预期审计。

## Artifact 验证

- deterministic USTAR SHA256：
  `b22827e6e2d8d33b03686fc177dc8f9c55c5133470fbe40f9fb3e33cce809fb5`；
- size：1,269,760 bytes；file count：102；
- tree inventory SHA256：
  `e7f0047b20c27bac1f4d61d7c0bff2e63c7760469cf9ccaa188dca948b7a4359`；
- fresh immutable download 与 source archive 逐 byte 相同；
- compact [`artifact.json`](artifact.json) SHA256：
  `77cfebc01909b1a018f7fe4ab300b843695ca9726fb96d3035587803bb4c2711`；
- compact [`summary.json`](summary.json) SHA256：
  `38c53dbaa587450ef09ddbd54c483c5f6120e8ce003cfca6fd8298280cad809e`。

runner 与 packager 的 exact 参数面见 [`../../../code/README.md`](../../../code/README.md) 和
[`../../../docs/restoration_v2_2_eager.md`](../../../docs/restoration_v2_2_eager.md)。
