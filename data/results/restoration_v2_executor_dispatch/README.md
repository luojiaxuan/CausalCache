# Restoration v2 AndroidWorld executor dispatch

## 结论

**通过：`PASSED_EXECUTOR_DISPATCH`。** Formal attempt
`rv2-20260715T101814Z-53016a40` 在 Aries 对冻结的 14 个 action cases 全部得到 exact HTTP 200
`JSONAction(...) executed.`；独立 reducer 从 raw records 重算得到 14/14。缺坐标 click 先由 pinned
`JSONAction` constructor 接受，随后 live actuation 按预期返回 HTTP 500，失败后的 health 与 cleanup reset
仍通过。

run 绑定 Git commit `b6e57c2619e88b8646657b3190bf45853a86c3d2`、MobileAgent revision
`11cea575561fb7800b5fb6b6cafa56f7a91de11f`、Aries runtime/server container ID、实际 image ID、port、mount
与四个 live executor source hashes。pre-inspection、53.39 秒 dispatch 和 post-inspection 的时间顺序完整，
前后 container/image/port/source identity 未变化。

## Claim 边界

该结果闭合 `native output -> parser -> canonical action -> AndroidWorld payload -> JSONAction -> live
execute_action` 的 action dependency。它不证明每个 case 都产生可见 screenshot change，不证明 policy
reference、restoration sensitivity、task success 或 CausalCache 方法收益。本次未加载 policy、未使用 GPU、
未生成 v2 policy output 或 restoration label。

## Evidence files

| 文件 | Bytes | SHA256 | 作用 |
| --- | ---: | --- | --- |
| `before-rv2-20260715T101814Z-53016a40.json` | 2,571 | `2cb7134351e2a526743ca11318526520822ae88cbdc9a99f6dad0acb5bd15f9a` | dispatch 前 live inspection |
| `raw-rv2-20260715T101814Z-53016a40.json` | 44,155 | `e1c801e56317f7638ffa9afc352d52152dcff979e82c0f27f4c61a837624587d` | runner raw transport records |
| `after-rv2-20260715T101814Z-53016a40.json` | 2,571 | `b36beafb0a9b4c251c93621ec3db7227cb6c4409e3a00daeaff2e23aacdb2321` | dispatch 后 live inspection |
| `summary-rv2-20260715T101814Z-53016a40.json` | 98,884 | `61956a457fb15a8fdfd35baf1a9f07c48ab45910a94829d537c9a591b2ffcc39` | immutable package 与 canonical verdict |

首个 live screenshot 为 1080x2400，transport body 35,977,173 bytes，SHA256
`8e4ac6ad79dd2f01cc70c2e66f113e23da1b8cbe200b6d815cd375cd6b20dda4`。pixels 未进入 Git；shape 与 digest
由 frozen runner 计算。compact action/reset/health bodies 保留 raw UTF-8，offline reducer 会重算 bytes、
SHA256、decoded JSON/text、case denominator、action-type counts 与所有 request 时间区间。

## 复核

```bash
make validate-restoration-v2-executor-dispatch
```

完整 producer 命令与失败处理见
[`../../../docs/restoration_v2_executor_dispatch.md`](../../../docs/restoration_v2_executor_dispatch.md)。
