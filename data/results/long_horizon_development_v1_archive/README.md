# Long-horizon development v1 归档

终态为 `SUPERSEDED_BEFORE_SCIENTIFIC_SCORING`。旧 `n=8/16` independent / conditional / v4 residual
实验不再执行；该终态不是 GO、NO-GO 或失败的科学实验。

已保留：24/18 split、24 条 development 的 screenshot/OCR/event/state substrate、repaired Source-A/Selection-B、
数据读取与验证代码，以及终止前已经完成的 576-record label-blind selector seal。后者仅作为 forensic evidence，
不能用于声称 selector 有效。

canonical artifacts：

- private HF dataset：`gavinlaw/causalcache-long-horizon-development-mobile`；
- repaired reusable substrate：`d237271e3266a72cce7aa730d0708936f1552365`；
- superseded archive：`5800f150f34d454ca72ae3eaeee3f30f564d834e`；
- archive tag：`long-horizon-development-v1-superseded-archive`；
- HF archive path：`archives/long-horizon-development-v1/`；
- archive payload manifest SHA256：
  `10d1cd251279c0c3e998254db43b2d829fbbcbe058457be8cfff41dbfd04c6e5`。

archive revision 的 4 个文件已 fresh-download，并与上传 staging bytes 逐 byte 对比通过。机器可读终止合同在
[`archive_payload_manifest.json`](archive_payload_manifest.json) 和 [`summary.json`](summary.json)。历史协议与
边界见 [`docs/long_horizon_development_v1.md`](../../../docs/long_horizon_development_v1.md)。

关键零计数：restoration distance、restoration label、policy generation、teacher forward、GPU KL、
closed-loop 与科学 GO/NO-GO verdict 均为 0。Hyper00 本地目录只作 cache/staging，不是 source of truth。
