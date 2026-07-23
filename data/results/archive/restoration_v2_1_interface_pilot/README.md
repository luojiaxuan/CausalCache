# Restoration v2.1 fixed-15 interface pilot

本目录保存 v2.1 official-tool native-output interface 的唯一 fixed-15 formal attempt 轻量证据。raw decoded
outputs、attempt/state records 与 runtime manifest 不进入 Git；它们作为一个 deterministic USTAR 保存在 private
Hugging Face dataset。

## 正式结果

- outcome：`PASS_V2_1_INTERFACE_PILOT`；
- status：`COMPLETED_FIXED_15_STATE_INTERFACE_PILOT`；
- source Git commit：`70724bdb82484cd5645d174765724bd1f83b5ace`；
- run-contract SHA256：`e4ff3b4237fa59a16af659253f1e96f326acc04b304f345c6f436614fd7272fc`；
- fixed denominator / attempted / completed：15 / 15 / 15；generation calls：15；
- strict whole-output parse：15/15；model-emitted closer：15/15；AndroidWorld bridge：15/15；
- retry、top-up、max-token truncation、surrounding prose、`Action:`、observation、第二 JSON/tool call：全为 0；
- teacher forward、KL、restoration label、expert action read、label-train/confirm generation：全为 0；
- UTC bracket：`2026-07-15T21:52:37.167342Z` -- `2026-07-15T21:54:01.446649Z`，耗时
  `84.279307` 秒；
- runtime：Hyper00 H200，container-local `cuda:0` / host physical GPU 2，UUID
  `GPU-e19275bf-adc5-9fc3-42d7-9a3d4b666b81`，BF16，PyTorch `2.11.0+cu130`，Transformers `5.6.0`。

GPU preflight 确认 8 张卡均无 compute process，pilot 只使用一张卡。utilization monitor 在 model loading、CPU
processor 与 frozen batch-1 generation 交错时观测到低于 90% 的窗口；已核查为单卡最小配置，且不能为本 gate
改变 batch-1 generation semantics。本结果不声称 throughput efficiency。

## Source of Truth

- private HF dataset：`gavinlaw/causalcache-restoration-v2-1-interface-pilot-mobile`；
- tag：`v2.1-interface-pilot-v1`；
- immutable revision：`bdff8ca71f150afd80d6291b4ecec76cbf9e7432`；
- path：`raw/restoration-v2-1-interface-pilot-v1.tar`；
- archive format/files/size：USTAR / 34 / 133,120 bytes；
- archive SHA256：`f71d5fd575dde48ae8b3e50a19dd2fbecfa02d7d5ae6087f909a47dfd7032064`；
- tree inventory SHA256：`095e0edcd01d5b85358d773fc28c22b7f4245e81b968c78b2551f8948f198df7`；
- Git binding：[`artifact.json`](artifact.json)，SHA256
  `e7a92a0469e3b66113158d5c2ce713c363ecaff97f95498f3ef84f02a4a6bed8`。

上传后已按 40-hex immutable revision 强制 fresh download；下载件与 formal archive 的 byte SHA256/size 完全
一致，HF tag 也解析到上述 revision。manifest commit/push 后，artifact validator 已在 clean descendant
`main@64581118eb3375a03234ef49e5ce1f7bc41d354e` 从该 fresh archive 返回
`VALID_RESTORATION_V2_1_INTERFACE_PILOT_ARTIFACT`、`archive_hash_verified=true` 与同一
`PASS_V2_1_INTERFACE_PILOT`；committed binding 已闭合。

## Claim 边界

该 PASS 只证明 frozen GUI-Owl policy 在 exact 15-state development denominator 上能通过 v2.1 native
official-tool output interface。它不证明两次 action agreement、finite teacher logits、memory sensitivity、
restoration gain、gate quality、terminal success 或论文主张；也不授权访问 confirm split。下一步必须先冻结
unchanged-interface full-45 source/contract，不能重跑、扩充分母或复用本 canonical attempt。这里的
AndroidWorld bridge 只验证 canonical action 到 executor action object 的转换，没有 emulator actuation 或 task
success 含义。
