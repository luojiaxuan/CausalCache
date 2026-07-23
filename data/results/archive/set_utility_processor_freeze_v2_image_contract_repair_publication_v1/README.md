# Processor v2 Immutable Publication Finalization

## 正式状态

```text
FINALIZED_PROCESSOR_V2_IMMUTABLE_HF_PUBLICATION
```

processor-v2 formal artifact 已通过 remote read-only validation、immutable revision 与 annotated tag
解析、两份 retained fresh replay 的逐 byte 复核。Git finalizer 本身没有执行 HF mutation。

## Immutable HF binding

- private dataset：`gavinlaw/causalcache-set-utility-new-development-mobile`；
- prefix：`artifacts/processor-freeze-v2-image-contract-repair`；
- annotated tag：`phase1-b2-processor-freeze-v2-image-contract-repair`；
- immutable revision：`c20bab8df424dc9e45ece1084f3d1dc035dd1ed8`；
- remote files：`25`；
- formal inventory SHA256：`7c2a971658ad9a6bbc639446e9326e9dd4718b8d2d77186f52a17f10f3b32fc1`；
- publication receipt SHA256：`ad69d2018e4f3d52a80647427edd67245e08a92796eadf50bb2e15fd30b46278`。

完整 validated receipt 与 Git source binding 见 [`summary.json`](summary.json)。原始 PENDING summary/card、
外置 `0600` receipt 和两份 fresh replay 均保持原字节与原路径，可继续使用 publication manager 的
`validate-only` 独立复验。
