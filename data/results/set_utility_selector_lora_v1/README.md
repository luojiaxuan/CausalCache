# Selector-side LoRA v1

状态：`PHASE1_COMPLETE_JOINT_ADAPTATION_RUNNING`。

- teacher/action policy：原始 frozen GUI-Owl，不含 LoRA；
- selector branch：LM top-4，计划 q/k/v/o rank-8 LoRA；
- Hyper00 单 context parity：sequence `[1,1522,4096]`，selected final tokens bitwise equal，max abs
  difference=`0.0`；
- 训练 labels：复用现有 900 trajectories / 9,287 optimizer states / 84,441 candidate-complete groups；
- 新数据需求：约 23,714 contexts 的 layer-32 full-sequence boundary cache，预计约 277GiB；
- boundary allowlist：23,714 contexts（query=9,543、event=14,171），content=
  `4172d46895edc14b507defe9af479cdf072e65fb026b7f2c79cb194de866c510`；
- boundary extractor/finalizer、LoRA trainer、token-adapter control 与 top-4 LoRA composite 已实现；
- Hyper00/Hyper01 两机共 12 个独立可恢复 partitions 已完成 23,714/23,714 contexts，0 failure。执行清单见
  [`boundary-rollout-plan.json`](boundary-rollout-plan.json)；
- Hyper00 单机 cache 已 finalize：3,065 shards、289,753,201,728 bytes、content SHA256=
  `77dec757c5637d538637463984caa5dff8481e36163589d540bccc5a7c55d16b`；本地路径
  `/data02/jaxan/runs/causalcache-selector-boundary-h00-c8cbc63`；
- executable training config 已绑定最终 cache SHA；token-adapter phase 1 checkpoint SHA256=
  `fa9d53bf...ce36e`；LoRA-only phase 1 已由 Hyper00 GPU 2--7、容器
  `sglang-omni-jaxan-07211541` 完成，checkpoint SHA256=`db95948b...ccc01`，code revision=
  `5c827e6852e8970fbcf2fb3cba0e3c53ae73fd2f`；
- token-adapter phase 1 truth 已完成并回填：B1/B2/B3/B4=
  `0.18320309/0.27833197/0.37139556/0.40280036`，macro/Long+=`0.30893275/0.30400585`；同一
  denominator 的 recent 为 `0.18320309/0.35564526/0.42122277/0.53311916`，macro/Long+=
  `0.37329757/0.38001933`。adapter-only 没有改善表示，但这不是 joint adapter 或 LoRA 的 verdict；
- token-adapter schedule family/status mismatch 在 truth 生成前被发现并修复；修复后 schedule content=
  `dee264a7...106ec`，物化为 132 states、3,163 coalitions（其中 3,031 forward）。Hyper01 6×H200 truth
  rollout 已完成：132/132 states、6/6 workers、0 skip/error，formal truth content=
  `ac7304d527af648da715f9045fbbf446268ef50c1634ae26ea09049741c2ae98`；phase summary SHA256=
  `f356809a44d1336c428e7ceaae1f32364297366aa9b98d63374bf9beb9239f98`。执行计划见
  [`token-adapter-truth-rollout-plan.json`](token-adapter-truth-rollout-plan.json)。
- LoRA schedule content=`90463d5f...d7f37`，物化为 129 states、3,785 coalitions（其中 3,656
  forward）。Hyper00 6×H200 truth rollout 已完成：129/129 states、6/6 workers、0 skip/error，formal
  truth content=`ab5f14cd0fc5ec403b271631681d1198069166240c34c31d50b61a49a8a6c8d0`。LoRA-only
  B1/B2/B3/B4=`0.18320309/0.28059072/0.35835215/0.39953378`，macro/Long+=
  `0.30541994/0.32770226`，低于 recent。执行计划见
  [`lora-truth-rollout-plan.json`](lora-truth-rollout-plan.json)。
- joint token-adapter epoch 1 checkpoint=`c50b4291...b51451`；91/91 states、2,251 coalitions 已完成，
  0 skip/error，truth content=`251da3dd...07544`。B1/B2/B3/B4=
  `0.18320309/0.24648697/0.26347535/0.26361162`，macro/Long+=`0.23919426/0.28640421`；因明显
  退化，该 control 不再增加 epoch。joint LoRA e1 checkpoint=`02e0ac96...f949`，201 states / 6,840
  coalitions（6,639 forwards）正由 Hyper00/Hyper01 共 11×H200 并行回填。执行记录见
  [`token-adapter-joint-truth-rollout-plan.json`](token-adapter-joint-truth-rollout-plan.json) 与
  [`lora-joint-truth-rollout-plan.json`](lora-joint-truth-rollout-plan.json)。
- boundary cache 完成后需上传 private HF；当前状态 `PENDING_HF_UPLOAD`。

设计与限制见 [`docs/set_utility_selector_lora_v1.md`](../../../docs/set_utility_selector_lora_v1.md)。
