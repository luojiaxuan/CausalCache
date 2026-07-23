# Restoration v2 AndroidWorld constructor preflight

## 结论

**通过。** 在 Aries 的 clean detached checkout 上，冻结 action fixture 的 14 个合法 payload 全部被
`X-PLUG/MobileAgent@11cea575561fb7800b5fb6b6cafa56f7a91de11f` 中真实
`android_world.agents.new_json_action.JSONAction(**payload)` 接受。

这只闭合 schema constructor，不等于 `/execute_action` 已把动作派发到 Android device。正式
device-side executor dispatch 仍是下一道独立依赖；本次没有加载 GUI-Owl、没有使用 GPU，也没有生成
policy output、restoration label 或方法效果结果。

## 不可混淆的状态

本目录 evidence 绑定的是 commit `a9e2afa` 中 run 前的 interface source snapshot；其 constructor 字段
刻意保留 `pending`。当前 manifest 可在 policy output 前因状态说明或后续 preflight source 更新而前进，不能
用当前工作树文件替代历史 bytes。本目录的 [`summary.json`](summary.json) 是随后在 exact manifest hash
`02744d82a91d0dca12d4c8792d78e6140e443cfbe674805dfe14d4665d9d017d` 上产生的 run evidence，字段为
`passed`。二者分别回答“运行前冻结了什么”和“冻结后实际运行是否通过”。

完整 host、container、Git、source revision、argv 与时间见 [`run_manifest.json`](run_manifest.json)。
