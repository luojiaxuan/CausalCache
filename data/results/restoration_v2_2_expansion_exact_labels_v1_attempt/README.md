# Restoration v2.2 expansion exact-label v1 attempt

本目录保存唯一 v1 formal attempt 的轻量 failure binding。该 attempt 已永久封存为
`INVALID_EXPANSION_EXACT_LABEL_ATTEMPT`，不能 resume、retry、top-up、删除后重跑或覆盖；任何后续修复必须使用
新的 protocol、attempt identity、source freeze 与 canonical output path。

## 结论

两个 H200 worker 均完成固定的 96-state shard，原始科学 payload 的独立 reducer 也得到
`PASS_V2_2_EXPANSION_EXACT_LABELS_V1`：192/192 states、1,792 条 $D(S)$、1,856 deployment edges、
3,072 full edges、1,984 interactions、576 attributions 与 192 exact oracles 全部命中；1,984 teacher forwards、
1,792 KL 与 1,792 scalar transfers 精确相等，repeat KL 最大值为 0，retry/top-up 与禁止操作均为 0。

但 formal attempt 的最终状态仍是 **INVALID**。source-locked monitor 共记录 3,850 个连续样本，覆盖两个 worker
启动前到结束后，索引 0--3849 无缺口；冻结契约要求相邻样本最多间隔 3.0 秒，实际有 5 个间隔超过阈值，最大
为 3.883721 秒，p99 为 2.292645 秒。因此 post-worker execution-evidence validator 正确 fail closed：

```text
ValueError: monitor did not cover the complete worker execution window at the frozen cadence
```

不能把内部 aggregate 的 PASS 写成 formal label PASS，也不能据此解锁 formal-58 gate training。当前只证明：完整
scientific bytes 已产生且 algebra/count/repeat gates 通过，但原预注册的 operational evidence gate 没有通过。

## 证据与保留状态

- source A：`2c00c9118dc00cc1bda361325d24d79d8c14f8b6`；
- execution head：`bccaab394ccbade6c9abf6281ab4d6e03820a362`；
- runner freeze：`c0447acda3f09bccc65461ed08092fa6b166370721767cf5f35eb19cc59583d6`；
- run contract：`63cddd712d9fa22bbd68110966161f36318d5fe1962b9b1477bae342da4180b8`；
- host / container：`hyper00` / `8fa31506ff8b274a304bc778b0fe5768112dbd3065c90e438d761a0e9a4154cc`；
- physical GPU 2/3，visible UUID 为 `GPU-e19275bf-adc5-9fc3-42d7-9a3d4b666b81` 与
  `GPU-f2d88c45-534f-b4cb-b9b9-65a9888c3617`；
- 开始 / 结束：`2026-07-17T06:57:02.359556Z` / `2026-07-17T08:01:07.769249Z`；
- global ledger SHA256：`77d3ad318a9c57e78a15edac0bb5273ade9ceeb95858cc6b33c731a6ee66a120`；
- aggregate SHA256：`b56126fa2dfd8b7a5ba40739abcc6cc36625e94352c8e420b2512a20f76d32fa`；
- monitor log SHA256：`da2c3281fd68236ccfae4dcb6d34a51d3dd1deb95e2c11fe725714195344cf4b`。

Hyper00 的原始目录
`/data/experiments/causalcache/restoration-v2-2-expansion-exact-labels-v1` 含 402 files / 3,579,535 bytes，
必须原地保留。canonical PASS archive 与原计划 HF repo 均未创建。invalid raw evidence 仍待 versioned、
no-overwrite 的 private HF archival；完成前该目录只是 staging，不是 source of truth。

## 下一步边界

下一步冻结独立的 **CPU-only child validation repair**：保持 v1 producer root、外部 `INVALID` ledger、内部
completed snapshot、worker ledgers 与 monitor bytes 逐 byte 不变；不再设置一个刚好容纳 3.883721 秒的新数值阈值，
而是把 cadence failure 永久保留为 original-attempt metadata，并只用 lifecycle coverage、连续索引、单调时间、
GPU UUID/source binding、独立 raw reduction 和 external-input replay 验证科学 payload。repair 只能声称
`scientific payload validated from an invalid monitor envelope`，不能追认 v1 formal PASS。

原始 invalid evidence 还要先进入独立 private HF forensic archive；该 archive 与 repair child 都不得占用原计划
formal PASS repo/tag/path。CPU repair 的独立 reduction、operation accounting、external replay 或 aggregate comparison
任一不一致，立即停止并改为全新 v2 identity 的完整 192-state GPU run；不得只补跑或挑选更好的版本。

不得事后把阈值调到 3.883721 秒，也不得把 v1 ledger 改写成 PASS。formal gate、fresh-16 evaluation、
matched-NLL、closed-loop 与 confirm 继续 locked；只有 repair child 的 immutable HF fresh replay 闭合后，才能显式
把该 child revision 接入 formal-58 gate。machine-readable 记录见
[`summary.json`](summary.json)。
