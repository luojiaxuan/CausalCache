# Policy-vision v3 validation repair v1

本记录是纯 CPU、只读的 validator repair audit。它只把 evaluated row 的七键 state 精确投影回
四键 feature state，并使用 producer `a935a3cf5efb1fa7a952ca6f45a5994609b367e9` 的冻结重建逻辑与
immutable labels/witness 逐 byte 重建原 exact-three artifact。没有重新运行 GPU、model、image processor、
vision feature、policy、generation、restoration attribution 或 gate；原 artifact 与原 GPU attempt ledger 未修改。

- status: `VALID_RESTORATION_V2_2_POLICY_VISION_V3_VALIDATION_REPAIR_V1`
- validation source: `dbb45637cf79c3573bbbc6051b8b480e3f76d69d`
- producer source: `a935a3cf5efb1fa7a952ca6f45a5994609b367e9`
- exact-byte matches: `True`
- producer scientific payload: `811e59c780851c19b7fe21314be1e52c1dbfcfa002e63a9857fd34d90ba94f48`
