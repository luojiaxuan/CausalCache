# Restoration v2 接口冻结

## 范围与状态

本文档补齐 scientific contract 留给实现层的确定性细节。它不改变
`causalcache_restoration_v2.json` 的 estimand、split、budget、distance 或 gate；这些规则由 scientific
config 的 raw SHA256
`9b9b78d9e1902d6ba7c648c939809c56fe55cccc17de58d4e6eed8d9ddf746cc` 单独锁定。

CPU interface fixture 已通过：14 个合法 action cases、23 个非法 cases、6,000 个完整标量坐标检查，以及
steps 4/5/6 共 28 个 coalition，其中 step-6 四候选为 16 个。该检查没有生成任何 policy output。后续 pinned
AndroidWorld `new_json_action.JSONAction` constructor 已有独立 14/14 run evidence，见
`data/results/restoration_v2_constructor_preflight/`；source manifest 中的 `pending` 是 pre-evidence sentinel，
不作为可变运行状态。device-side executor dispatch 仍为 pending，因此完整的
`prompt -> parser -> bridge -> executor` dependency 尚未闭合。

## Action 与 canonical target

v2 使用独立模块 `causalcache.policy.gui_owl_v2`，不修改历史 v1 的 `gui_owl.py`：

- canonical action 必须精确使用小写
  `click/long_press/swipe/type/system_button/open/wait/answer/terminate`；
- 只额外接受精确 alias `tap -> click`、`open_app -> open`，不做 action name casefold；
- button 必须精确为 `Back/Home/Enter`；`key`、`Menu`、`time` 与 `terminate:failure` 均拒绝；
- 每种 action 的参数集合必须与 scientific config 完全相等，missing、extra、错误类型、duplicate JSON key、
  `NaN/Infinity`、额外 prose 或 thinking block 全部 fail closed；
- coordinate 必须是两个原生 JSON integer，bool/float/string 均拒绝；完整 `0..999` 标量域按
  `floor(value*(extent-1)/999+0.5)` 检查端点、单调和不越界；
- text 只做 Unicode NFKC，保留大小写、空白和空字符串，不做 strip、casefold 或 truncation；
- alias 在 archive 中保留 raw native tool-call，但 canonical action、teacher target 和 strong summary 全部使用
  alias-normalized action name；
- teacher target 固定为非语义 carrier
  `Action: Execute the selected mobile action.\n` 加 canonical compact `<tool_call>`，generated Action
  description 不进入 teacher-forced target。

合法 fixture 位于 `data/fixtures/gui_owl_v2_action_roundtrip.json`。本地 validator 会把 canonical action
映射为实际 AndroidWorld payload；远端 integration preflight 还必须把这 14 个 payload 全部传给 pinned
`JSONAction(**payload)`。

## Strong low-fidelity event

v2 object 与 serialization 固定为：

1. 八个 key 按 scientific config 顺序写入；
2. `screen_text_added` 与 `screen_text_removed` 是 ordered JSON `list[str]`，无变化为 `[]`；
3. `unknown` 只用于缺失 scalar，不用 `unknown` 冒充空 text delta；
4. compact `ensure_ascii=false` JSON 后恰好一个换行，再编码 UTF-8；
5. artifact 同时保存 object、exact serialized string 与 SHA256；prompt 前重算并逐字节验证。

实现层的其余歧义冻结如下：

- text backend 输出先按 bounding box `(top,left,bottom,right,normalized_text)` 排序；每个 node 做
  NFKC、collapse whitespace、strip，再按 Unicode whitespace 切分；node 内保持 token 顺序；
- added/removed 使用带重复计数的 ordered multiset difference，各保留前 32 tokens；discarded count 只进
  artifact metadata，不进入 policy-visible summary；
- 多 app trajectory 的全局 `apps` 列表不能猜测 foreground。只有 per-event source app label 才算
  `source_app_label`，否则使用 executor package，再否则 `unknown`；
- spatial argument 为 `coordinate_bin:x{x*10//1000}_y{y*10//1000}`；
- swipe 先计算 finger endpoint 减 start 的 `(dx,dy)`，用 Chebyshev displacement
  `max(|dx|,|dy|)`；vertical 在 axis tie 时优先。summary 写 viewport navigation direction，即 finger
  方向的反方向：finger 向上写 `viewport_down`；displacement `1..332/333..665/666..999` 分别为
  `short/medium/long`，zero swipe 写 `viewport_stationary:zero`；
- `type/open/answer` argument 为 exact NFKC text，system button 为 exact button，wait 为 `wait`，
  terminate 为 `success`；
- `screen_change` 的 RGB MAD 只接受已由 pinned 256x256 bilinear backend 产生的 RGB pixels；本地纯逻辑
  计算 MAD 与 0.005/0.05/0.20 bins。实际 Pillow/resampler identity 仍须写入 execution config；
- executor result 只从显式 executor acceptance record 得到 `accepted/failed`，缺失为 `unknown`。

## Post-state-only prompt

derived artifact 的每个 event 使用新字段 `low_fidelity_v2`、`low_fidelity_v2_serialized`、
`low_fidelity_v2_sha256`，不覆盖 parent artifact 的历史五字段 `low_fidelity`。prompt builder 接受一个显式
trajectory ID 和 decision step：

- system + 单一 user message，不伪造 assistant history；
- 每个 history event 始终写入相同 `Event summary` text block；
- 若 event 属于 coalition，只在该 summary 后插入它的一张 `observation_after` image block，不增加任何
  conditional text、before image、source tool call 或 action description；
- 最后恰好加入一次 current observation；latest current-equivalent event 始终 summary-only；
- builder 不强制 `B=2`，因为 summary-only、reference `S={1,2,3,4}` 与 attribution coalitions 共用同一
  API；budget 由上层 frozen runner 约束；
- candidate/current identity 使用 artifact member path 与 SHA256；builder 先读取原始 image bytes 并核对
  archived SHA256，再把通过校验的 bytes 交给 decoder。

fixture `data/fixtures/restoration_v2_prompt_low_fidelity.json` 穷举 steps 4/5/6 的 28 个 coalition。每个
decision state 内抽掉 image blocks 后，所有 coalition 的 policy-visible text list 必须逐字相同；image
必须紧随对应 event summary，image count 必须为 `1+|S|`。step-6 reference 为 5，summary-only 为 1。

## 验证与下一道门

本地 CPU 验证：

```bash
make validate-restoration-v2-interfaces
```

pinned AndroidWorld constructor preflight 必须在 source 内显式传入路径，并将轻量 summary 写入 Git：

```bash
cd code
python3 -m scripts.validate_restoration_v2_interfaces \
  --contract configs/causalcache_restoration_v2.json \
  --action-fixture ../data/fixtures/gui_owl_v2_action_roundtrip.json \
  --prompt-fixture ../data/fixtures/restoration_v2_prompt_low_fidelity.json \
  --interface-manifest ../data/manifests/restoration_v2_interfaces.json \
  --androidworld-source-root /data/<PINNED_ANDROIDWORLD_SOURCE> \
  --androidworld-source-revision <FULL_40_HEX_GIT_SHA> \
  --container-image-digest sha256:<64_HEX> \
  --run-git-commit <FULL_CLEAN_CAUSALCACHE_GIT_SHA> \
  --output-summary ../data/results/restoration_v2_interface_preflight/summary.json
```

本地 validator run summary 的 `androidworld_json_action_constructor_validation.status` 必须从 `not_run`
变为 `passed`，并记录 source revision、container digest 与 Git interface manifest。冻结 source manifest
在该 preflight 前用 `pending` 表示尚无 constructor evidence；两者不是同一个状态字段。即使 constructor
通过，device-side executor dispatch 仍须单独验证。该 preflight 仍是 CPU/action substrate，不允许加载
GUI-Owl 或生成 policy output。
