# Long-history oracle ceiling diagnostic v1

状态:`COMPLETED_SET_UTILITY_LONG_ORACLE_EVALUATION / HEADROOM_CONFIRMED`。

250 个 train long/very-long states(220 long + 30 very_long,158 trajectories,`n_t` 17--43)上,
wave 式 candidate-complete singleton + true-greedy restoration oracle 与 recent/additive/random 的
trajectory-equal normalized recovery。四个 wave 均 250/250 completed、0 skip,跨 wave reference
action 漂移 0。总计 25,032 个 coalition 标签(7,654 + 5,904 + 5,833 + 5,641)。

| method | B1 | B2 | B3 | B4 | macro | very_long macro |
|---|---:|---:|---:|---:|---:|---:|
| oracle_greedy | 0.5279 | 0.6611 | 0.7368 | 0.7851 | **0.6777** | 0.6358 |
| additive top-B | 0.5279 | 0.5035 | 0.5551 | 0.5322 | 0.5297 | 0.4797 |
| recent | 0.0325 | 0.2827 | 0.3282 | 0.1981 | 0.2104 | -0.4183 |
| random | 0.1312 | 0.2171 | 0.2390 | 0.3040 | 0.2229 | 0.0876 |

- `oracle_greedy - recent` macro = `+0.4674`,trajectory-clustered paired bootstrap 95% CI=
  `[0.3591, 0.6304]`;预注册判定阈值 `lower_95 > 0.05` 以约 7 倍裕量满足 → **HEADROOM_CONFIRMED**。
- `additive - recent` macro = `+0.3193 [0.2360, 0.4306]`,但 additive 在 B2--B4 平台化
  (0.50--0.55),远低于条件贪心(0.66--0.79):长历史上纯 singleton-additive 选择显著不足,
  条件边际结构不可省略。
- recent 在长历史上整体接近 random(macro 0.210 vs 0.223),B4 反而低于 B3;very_long 上 recent
  macro 为 **负值**(-0.418)——只恢复近端事件而缺老上下文会主动破坏行为重建。
- best singleton 落在 recent-4 之外的比例 58.4%,平均 age fraction 0.363;长历史 oracle B4(0.785)
  与小历史 exact(0.825)同量级,restoration 信号在长历史上没有衰减。

## Artifact

- 本目录 [`summary.json`](summary.json):reducer 完整输出,content SHA256=
  `179b0d85e0ad0315efa88432eaccb77a5231b8c0ee7066739de1a3f78389cf9c`;
- 完整 payload(4 个 wave 的 schedules + 两 host 全部 state terminals + config):Hyper00
  `/data02/jaxan/artifacts/causalcache-long-oracle-v1-179b0d8`,`payload.tar.gz` SHA256=
  `9238f63dca61d7b18377a5df03448c2b15484522cb0e4a60cb32ffa9170c8b34`;状态 `PENDING_HF_UPLOAD`
  (目标 `gavinlaw/causalcache-set-utility-variable-history-mobile` path
  `artifacts/set-utility-long-oracle-v1-179b0d8`、tag `set-utility-long-oracle-v1-179b0d8`;
  本次会话的自动上传被权限分类器拦截,staged 内容与 README/summary 已就绪,可直接
  `huggingface_hub.upload_folder` 后打 tag);
- 原始 run roots:Hyper00/Hyper01 `/data02/jaxan/runs/causalcache-long-oracle-w{1,2,3,4}-labels-912e5d1`
  与同名 `...-schedules-912e5d1`;
- source revision `912e5d1b5ffddd1547c79da6cb3dc47b188ea8e1`,config SHA256=
  `a8b8b7dbb85d50fbd565c953f066bd274862dccff7202ac707015009658fb5c2`;
- labels 为 train-only,合同显式允许复用为 decision distillation v2 的 candidate-complete
  supervision;tune truth、untouched evaluation、policy replay、closed-loop 均未访问。

复现:[合同](../../../docs/set_utility_long_history_oracle_v1.md);
schedule/评测入口 `materialize_set_utility_long_oracle_schedules.py` /
`evaluate_set_utility_long_oracle.py`。
