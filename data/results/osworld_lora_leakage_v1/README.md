# OSWorld mobile-LoRA leakage v1

## Verdict

`odyssey_terminal_s60_alpha32` 正式通过预注册 gate：30/30 parser valid、0 hard mobile leakage，且 single-image
与 recent-B4 都是 15/15 valid。结论是 terminal-repair policy LoRA 没有把 OSWorld `computer_use` 输出拉回
`mobile_use` grammar；从语法迁移角度，后续 OSWorld closed-loop 可以使用该 LoRA，不需要退回 frozen policy。

早期 `odyssey_s75` 两种强度均为 29/30 valid，按门槛属于 mild leakage。唯一失败是同一个 Calc prompt 的
normalized coordinate 越界：alpha32 输出 `[1000,17]`，alpha16 输出 `[1005,17]`。它不是
`mobile_use`/mobile-only action 渗漏，而且 alpha16 没有修复。混入 terminal states 续训到 s60 后该 prompt
恢复合法。

| Profile | Parser valid | Hard leakage | Raw action changes | Normalized changes | JS vs frozen | Verdict |
|---|---:|---:|---:|---:|---:|---|
| frozen | 30/30 | 0 | -- | -- | -- | reference |
| s75 α32 | 29/30 | 0 | 4 | 2 | 0.01323 | mild |
| s75 α16 | 29/30 | 0 | 1 | 2 | 0.01323 | mild；未修复 |
| terminal-s60 α32 | **30/30** | **0** | 6 | **1** | **0.00184** | **PASS** |

## Desktop 实际发生了什么

terminal-s60 的 6 个 raw action changes 中，5 个只是 `click ↔ left_click` spelling 变化，经过 runner
normalization 后动作语义不变。唯一的 normalized action change 是一个 Calc prompt 从 frozen `wait` 变成
`left_click`。整体 action counts 从 frozen 的 `click/type/wait=26/1/3` 变为 `27/1/2`，没有出现
`right_click/double_click/drag/scroll/key`，也没有任何 `mobile_use/swipe/long_press/system_button/open`。

这证明 grammar 没有物质劣化，不证明 click 比 wait 更正确；action quality 仍必须由 OSWorld closed-loop
evaluator 判定。30 prompts 也只是便宜的迁移 gate，不能取代完整 benchmark。

| Input form | terminal-s60 valid | Hard leakage | Normalized actions |
|---|---:|---:|---|
| single image | 15/15 | 0 | click 13, type 1, wait 1 |
| recent-B4 five-image | 15/15 | 0 | click 14, wait 1 |

## Runtime note

同机独立 GPU 上，frozen / terminal-s60 generation mean=`0.951/1.220s`，seconds/token=
`0.0298/0.0379`。当前 evaluator 使用训练代码的 float32 forward hooks 注入 LoRA，没有 merge/fused adapter；
约 28% 的 mean overhead 是该研究实现的测量值，不应外推为部署下界。两者 peak allocated HBM 都约
`17.3 GiB`。

## Denominator 与边界

- official no-GDrive 361-task roster 等距取 30，覆盖 10 domains；
- 30/30 KVM fixture reset + WAIT 完成，0 failure；
- 15 single-image + 15 synthetic recent-B4。B4 在同一 live episode 的 initial/post-WAIT screenshots 之间
  交替，只测五图 prompt grammar，不作为 memory-quality evidence；
- greedy decoding，480 effective visual tokens/image，`max_new_tokens=256`；
- 四个 profiles 使用相同 prompt manifest 和 H100 runtime；
- 第一次 launcher 在任何 generation 前因 host absolute path 与 container mount path 不同而 fail-fast，0
  records；正式 attempt 改用 relative fixture paths，保留在 `profiles-r1`，没有追认失败 attempt。

## Source of Truth

- execution Git revision：`c53f2719ad32313e9f83e626377d7f73b00c1aa0`；
- upstream handoff：`1454181f5bb32c080941583322210b59ccaed82d`；
- config SHA256：`d7983fc50ce62aa2b5e608ffc6d98c586175d3dff3bf3a99ed64e98c9fe316b3`；
- prompt manifest SHA256：`78e528b2bac71b638e93c20ce190338ee9b9af45f6e107955f404a6f371587f0`；
- s75 checkpoint SHA256：`90a56b7e566552713f1e7adad0b20f23170c2a034eed338a795e8bf63628d56e`；
- terminal-s60 checkpoint SHA256：`cba455cbfede5e091c06d8714ef000d5582adb46c4a5343fa8c0ab8c4ab47b18`；
- reusable adapter destination：intended private HF model repo
  `gavinlaw/causalcache-gui-owl-policy-lora-mobile`，当前 `PENDING_HF_UPLOAD`；canonical publication/revision 尚未
  产生，Hyper00 source paths 与 H100 copies 只是 persistent staging/cache；
- model：`mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`；
- OSWorld：`b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf`；
- raw root：H100 `/data/jaxan/osworld-lora-leakage-v1/`；formal profiles=`profiles-r1`；
- machine-readable result：[`summary.json`](summary.json)，SHA256=
  `a9cf624de90f85973e65457ea8c59e8c85e68c223b525cfbe780f4e53ff78d31`；
- raw screenshots/outputs 为 host-specific diagnostic artifact，状态 `LOCAL_INFRASTRUCTURE_ARTIFACT`，无需上传
  Hugging Face；两个 reusable LoRA checkpoints 则保持 `PENDING_HF_UPLOAD`。Git 保存 config、代码、轻量
  summary 与完整 provenance。
