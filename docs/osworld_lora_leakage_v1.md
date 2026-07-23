# OSWorld mobile-LoRA leakage v1

## 问题

AndroidWorld/GUI-Odyssey 的 policy LoRA 只见过 `mobile_use` action grammar。该 LoRA 的权重对所有 language
model layers 全局生效，因此在 OSWorld 的 `computer_use` prompt 下可能保持 frozen desktop 能力，也可能把
输出拉回 mobile tool/action vocabulary。本实验只判定这种 grammar/action-distribution leakage，不测 task
success。

上游 handoff 固定在 `1454181`。测试实现不改变 frozen GUI-Owl action policy、LoRA checkpoint 或 OSWorld
parser，只增加 raw greedy output 的可审计接口。

## 冻结 denominator

- OSWorld revision：`b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf`；
- roster：official `test_nogdrive.json` 361 tasks；
- 从 roster 位置 `[0, 360]` 等距取 30 tasks，固定覆盖 10 domains：Chrome 4、GIMP 2、Calc 4、Impress 4、
  Writer 2、multi_apps 7、OS 2、Thunderbird 1、VLC 2、VS Code 2；
- 30 个 task 均通过真实 KVM reset 采集 screenshot，再执行一次固定 WAIT；0 fixture failure；
- 偶数 prompt 使用 single current screenshot，奇数 prompt 使用 recent at-most-B4：四个恢复 event images 加
  当前 screenshot。B4 history 在 live initial/post-WAIT 两张图之间交替，只用于测试五图 prompt 下的 grammar
  leakage，不作为 memory quality 或 long-horizon evidence；
- 四个 profile 必须消费逐字节相同的 manifest，greedy decoding、480 effective visual tokens/image、
  `max_new_tokens=256`。

## Profiles

1. `frozen`：原始 GUI-Owl-1.5-8B；
2. `odyssey_s75_alpha32`：Odyssey s75，rank 16 / alpha 32；
3. `odyssey_s75_alpha16`：相同 checkpoint，加载时 alpha 16；
4. `odyssey_terminal_s60_alpha32`：从 s75 继续混入 3,233 个终止态训练至 step 60，rank 16 / alpha 32。

LoRA 目标固定为全 language-model layers 的 `q/k/v/o`，vision tower 不注入 LoRA。s75 checkpoint SHA256 已
冻结为 `90a56b7e566552713f1e7adad0b20f23170c2a034eed338a795e8bf63628d56e`。本合同提交时 s60 仍在训练，
必须在任何 profile generation 前补入源/目标双端一致的 SHA256；不得用 s20/s40 替代或根据输出选 step。

## 指标与判定

每个 profile 报告：

- normalized OSWorld parser 合法率；
- raw 与 normalized action counts，以及相对 frozen 的 Jensen--Shannon divergence；
- raw `click` / `left_click` counts；
- single-image 与 recent-B4 分层；
- hard leakage：tool name 为 `mobile_use`，或 action 为 `swipe/long_press/system_button/open`。

相对 frozen 的预注册门槛：

- parser 合法率降幅 `<2` percentage points 且 hard leakage 为 0：`PASS_NO_MATERIAL_LEAKAGE`；
- 降幅 `[2,10]` points 且 hard leakage 为 0：`MILD_REQUIRES_ALPHA16`；
- 降幅 `>10` points 或任一 hard leakage：`FAIL_HARD_LEAKAGE`。

若 s75 α32 为 mild，只有预先列出的 s75 α16 可以作为强度修复。s60 α32 若为 mild，必须另立版本化 α16
profile，不能用未运行的结果追认通过。任何 profile 的 OSWorld task success、closed-loop 或 memory benefit 均
不由本实验解锁。

## 正式结果

terminal-s60 α32 获得 30/30 parser valid、0 hard leakage；single-image/recent-B4 分别 15/15、15/15，
相对 frozen 的 normalized action JS divergence=`0.001835`，按预注册门槛为
`PASS_NO_MATERIAL_LEAKAGE`。s75 α32/α16 均为 29/30，唯一错误是同一个 Calc prompt 的 coordinate 超出
`[0,999]`；两者都是 mild，α16 没有 rescue。完整解释与数字见
[`data/results/osworld_lora_leakage_v1/`](../data/results/osworld_lora_leakage_v1/README.md)。

该 PASS 只解锁“terminal-s60 LoRA 可进入 OSWorld closed-loop”，不解锁 OSWorld success、memory benefit 或
跨平台泛化 claim。

## Artifacts

- config：`code/configs/causalcache_osworld_lora_leakage_v1.json`；
- H100 fixture root：`/data/jaxan/osworld-lora-leakage-v1/fixtures`；
- manifest：`/data/jaxan/osworld-lora-leakage-v1/prompt-manifest.json`；
- raw profiles：`/data/jaxan/osworld-lora-leakage-v1/profiles-r1`；
- Git lightweight result：[`data/results/osworld_lora_leakage_v1/`](../data/results/osworld_lora_leakage_v1/README.md)。
