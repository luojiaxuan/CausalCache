---
pretty_name: CausalCache Restoration v2.2 Label Expansion Substrate v1
---

# CausalCache Restoration v2.2 Label Expansion Substrate v1

这是 CausalCache 192-state label-expansion reference substrate 的私有原始证据包。它只验证冻结
GUI policy 的 reference/summary behavioral substrate，不包含 restoration labels、gate checkpoint、
matched-NLL 或 closed-loop 结果。

## 身份与来源

- Git repository：`https://github.com/luojiaxuan/CausalCache`
- runner source commit：`1a3833d6951c768ce1bdd5f976d1044c291d002e`
- execution commit：`642feb28b4f7ce4e7bf9f7791f7fb0f6919c1839`
- frozen config SHA256：`42144f33e2473c787b3648c0ace18b22902fa3376615732aef8fd3aa5780a6e5`
- run-contract SHA256：`b467113130da74d943a00c2f11d7cad242204e475b7b806e76f666e670a1ed6a`
- HF tag：`v2.2-label-expansion-substrate-v1`

## 文件与校验

- payload：`raw/v2.2-label-expansion-substrate-v1.tar`
- format：deterministic POSIX USTAR
- archive SHA256：`4e77a38be34cb2f3c084a13abd47c0530e6729ff6cca72793977c62fa78ff47d`
- archive size：`7,475,200` bytes
- tree inventory SHA256：`56f291053121ccf813698a48beb6269b1fa1d096b7e974f6eb9424f55bc45543`
- archived file count：`406`

证据包包含 `run_manifest.json`、`aggregate.json`、`execution_evidence.json`、全局 attempt ledger、
两 worker 的 runtime/canary/attempt/state/terminal records，以及 GPU monitor evidence。consumer 必须先用
仓库中同 revision 的 artifact validator 独立重算 raw gate，不能只信 `aggregate.json`。

## 正式结果

- outcome：`PASS_V2_2_LABEL_EXPANSION_SUBSTRATE_V1`
- fixed denominator：`192/192` states，`0` failed
- operation counts：`384` generation、`576` teacher forward、`384` KL measurement
- parse、finite-logit、exact-repeat agreement：全部 `1.0`
- mean repeat KL：`0.0`
- memory-sensitive states：`185/192`（`96.3541667%`）
- retry、top-up 与 forbidden operations：全部 `0`

## 生成与验证

```bash
PYTHONPATH=code python3 -m scripts.manage_restoration_v2_2_expansion_substrate_artifact \
  package \
  --raw-output-dir /data/experiments/causalcache/restoration-v2-2-label-expansion-substrate-v1 \
  --global-attempt-ledger /data/experiments/causalcache/.restoration-v2-2-label-expansion-substrate-v1.attempt.json \
  --output /data/experiments/causalcache/restoration-v2-2-label-expansion-substrate-v1.tar \
  --source-git-commit 1a3833d6951c768ce1bdd5f976d1044c291d002e \
  --config-sha256 42144f33e2473c787b3648c0ace18b22902fa3376615732aef8fd3aa5780a6e5
```

上传后必须从 immutable HF commit fresh-download 到另一条本地路径，并运行
`validate-fresh-hf`；source archive 与 fresh archive 必须逐 byte 相同。
