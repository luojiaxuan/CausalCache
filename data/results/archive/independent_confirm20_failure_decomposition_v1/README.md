# Independent confirm-20 failure decomposition v1

本目录只保存 Git-reviewable 的轻量结果。20×16 `D(S)` 父表仍以 private HF parent report commit
`a0b408e58d629299be334a74ecbd0ec2fa2ed1fc` 为 canonical source；20-state derived rows、完整 aggregate report
与 bundle manifest 位于 private HF child
`gavinlaw/causalcache-independent-confirm20-failure-decomposition-mobile@31aa5e22c08a3d56bc729fcbc88d790cce4249c0`，
tag `independent-confirm20-failure-decomposition-v1`。

正式结果是 `CASE_A_ORACLE_INDEPENDENT_LOSES_TO_OCR_RGB`。raw utility sum 为：exact `0.856431`、
OCR/RGB `0.783268`、oracle-independent `J=0.765311`、learned independent `I=0.703385`。因此
`J-OCR` raw mean=`-0.000898`，paired 90% interval=`[-0.006370,0.004145]`，support=`10/20`；当前
independent projection 的 ceiling 本身没有胜过 strongest heuristic。

同时 `J-I` raw mean=`+0.003096`，paired 90% interval=`[+0.000477,+0.006533]`，说明 student 确有
distillation/generalization gap。但即便把 student 提升到 `J`，仍不足以跨过 OCR/RGB。因此结论不是“student
完全没问题”，而是“student 与 objective projection 两层都有损失；student 不是唯一 blocker”。normalized
口径下 `J` 高于 OCR/RGB `+0.033558`，但它在冻结合同中只作描述，不能翻转 raw-primary 路线。

Source-A=`d3451db74d7bdd415455b87a91718bbedfcd0884`；唯一单文件 Execution-B=
`9f20e4b55c5e69ae1b1c26d58cd7c52bdd5e0af3`。Hyper00 无 GPU container 完成正式 run；独立 validate 从 child
commit/tag fresh replay exact-three，remote mutation=`0`、local write=`0`。父
`NO_GO_INDEPENDENT_CONFIRM` 不变，closed-loop、matched-NLL 与 sealed AndroidWorld test 保持未授权、未执行。

完整数字、argv、runtime、pre-semantic failures 与逐文件 identity 见 [`summary.json`](summary.json)。
