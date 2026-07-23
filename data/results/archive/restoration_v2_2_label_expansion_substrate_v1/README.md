# Restoration v2.2 label-expansion substrate v1

本目录只保存 192-state expansion substrate 的轻量、可审计结果。完整 406-file raw evidence 不进入 Git，
其唯一 source of truth 是私有 Hugging Face dataset：

- repo：`gavinlaw/causalcache-restoration-v2-2-label-expansion-substrate-mobile`
- tag：`v2.2-label-expansion-substrate-v1`
- immutable revision：`25ac19cf6ef98adc243d421cd0039ac104ddb539`
- path：`raw/v2.2-label-expansion-substrate-v1.tar`
- archive SHA256：`4e77a38be34cb2f3c084a13abd47c0530e6729ff6cca72793977c62fa78ff47d`
- tree SHA256：`56f291053121ccf813698a48beb6269b1fa1d096b7e974f6eb9424f55bc45543`

正式唯一 attempt 从 clean pushed execution commit
`642feb28b4f7ce4e7bf9f7791f7fb0f6919c1839` 运行，runner source commit 为
`1a3833d6951c768ce1bdd5f976d1044c291d002e`。结果为
`PASS_V2_2_LABEL_EXPANSION_SUBSTRATE_V1`：192/192 states 完成、0 failed，384 generation、576 teacher
forward 与 384 KL measurement 精确命中冻结计划；parse、finite logits 和 exact repeat agreement 均为
1.0，mean repeat KL 为 0，185/192 states 对 summary/reference memory fidelity 敏感。

processor canary 在任何 generation 前通过 192 included mutations、128 excluded states 与 192
excluded-event mutations。两 worker 各完成固定 parity 的 96 states，retry、top-up 与 forbidden operations
均为 0。

source-locked GPU monitor 从 preclaim 前启动，覆盖模型加载、CPU-heavy processor canary、state 间隙与真实
forward；因此 2,368 samples 中 803 个低于 90% 的 incident 原样保留，不能只把它解释为 generation
利用率。外部连续 10 秒监控显示进入 state kernel 后多数窗口为 90%--100%，但该观察不替代 raw monitor
evidence。

[`artifact.json`](artifact.json) 是由 source archive 与 immutable fresh download 共同生成的机器可读绑定；
[`summary.json`](summary.json) 是团队交接摘要。fresh download 与 source archive 已逐 byte 相同，并由同一
artifact validator 独立重算 raw gate。

这个 PASS 只解锁 expansion restoration-label source 工作；当前仍没有 expansion labels、formal gate、
matched-NLL 或 closed-loop success，confirm split 保持未访问。
