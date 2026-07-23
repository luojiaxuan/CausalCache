# Restoration v2.2 policy-vision baseline v3 SizeDict interface repair

本结果完成 frozen GUI-Owl policy-vision feature-only comparator。15 条 primary `n=4,B=2`
train/development states 只把 event 1--4 的 post-action screenshot 与 decision-step-6 current screenshot
送入 image processor 和 final main vision merger；feature workers 从未接收 goal、text、OCR 或 restoration
labels。两个 H200 worker 分别固定到 `cuda:0`/`cuda:1`，并通过全部 same-device replay 与一条
cross-device sentinel replay。language-model/policy forward、LM head、generation、teacher/KL、gate、
matched-NLL、closed-loop 与 confirm/test access 全部为 0。

| Split | States | Mean recovery | At-most-B exact match | Exact-B match |
| --- | ---: | ---: | ---: | ---: |
| Train | 10 | 0.535228 | 0.300000 | 0.300000 |
| Development | 5 | 0.143615 | 0.000000 | 0.000000 |
| Overall | 15 | 0.404691 | 0.200000 | 0.200000 |

预注册 outlier `0141544666483837` 的 recovery 为
`-2.641163`；负值若出现会原样保留，不 clamp、不删除。

- source commit: `a935a3cf5efb1fa7a952ca6f45a5994609b367e9`
- contract SHA256: `794474d8bc60463ba10fdd772691461f5910ca5e5501f7cccff4c53542b84b7f`
- scientific payload SHA256: `811e59c780851c19b7fe21314be1e52c1dbfcfa002e63a9857fd34d90ba94f48`
- state score rows: `15` (`8c5977260bb1a8618b43cd877d963bd637c197e589db532c7ccbcab9c55f58b9`)
- worker GPUs: `GPU-e10a085b-e20e-a454-6fda-17063cf68418`, `GPU-2396c1ff-41ff-385f-30fe-0272d99c5d6c`
