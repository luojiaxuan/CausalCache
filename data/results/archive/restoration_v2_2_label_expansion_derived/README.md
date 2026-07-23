# Label-expansion derived artifact

本目录保存 48/16 gate data expansion 的 policy-blind derived artifact 轻量完成证据；可复用大文件位于 private
Hugging Face dataset，不进入 Git。

- HF repo：`gavinlaw/causalcache-guiodyssey-restoration-v2-mobile`
- tag：`restoration-v2-label-expansion-v1.0.0`
- immutable revision：`630363a6adb692d72774f16dd0653a50216313ff`
- payload prefix：`derived/restoration-v2-label-expansion-v1`
- inventory：64 trajectories、320 events、192 decision views、384 images、384 OCR records
- exact-six bytes：347,902,778
- tree SHA256：`9394b369e2b741e6aacf9ece4fc5dae3e6337b25a7e65402307e4e7862b94abc`
- OCR aggregate：`f3423ece941224706406d4bc7e1f1eac7f2f6512616a0368c0cce37a235ec2ff`

Builder、standalone pre-upload validator 与 immutable fresh-download validator 均通过。Builder 单独完成 16 个
source Parquet rehash 和 64-row reload；后两次 validator 只复核 artifact bytes、冻结输入与 OCR replay。

`artifact.json` 嵌入完整 payload manifest、exact-six 逐文件 witness、执行 metadata 与 negative declarations。
本阶段没有 policy forward、restoration label、gate checkpoint、matched-NLL、closed-loop 或 confirm access。
