# Restoration v2 immutable parser compatibility replay

本目录保存对第一次 45-state substrate raw trace 的正式、离线、不可变 replay。它从 clean pushed
`main@fc3adf13d48bb016014f7efa62bd27c8a4d12f49` 运行，审计前后均确认本地 `HEAD`、`origin/main` 与
GitHub remote `main` 相同，并把实际 import 的四个 Python source 逐字节绑定到该 commit。

## 正式结论

结论是 `NO_GO_ADAPTER_ONLY`：

- 原 strict parser：0/45；
- 恰好一个 `Action:` 且首个 balanced JSON 可 canonicalize：43/45；
- 保守 compatibility recovery：40/45，即 88.89%；
- frozen gate 为 0.99，45-state denominator 实际要求 45/45，因此 adapter gate 不通过；
- 40 个 accepted 中只有 25 个在首个 JSON 后 clean EOF；另外 15 个需要丢弃 exact-whitelist suffix：
  重复 `<tool_call>` opener 8、extra brace 4、extra brace + opener 3；
- model-emitted canonical `</tool_call>`：0/45；
- 其余 5 个明确拒绝：多 action 1、截断 JSON 1、第二 JSON 1、额外 observation 2。

所以 40/45 不是 native well-formed parse coverage，也不能用事后 parser 覆盖原 `NO_GO_V2_SUBSTRATE`。
它只说明大多数输出含有可恢复的 action payload；仅换 parser 仍不足以越过原 gate，下一步必须另立 v2.1
generation interface。

## 不可变输入与审计边界

- private HF dataset：
  `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile@restoration-v2-substrate-screening-v1.0.0`；
- immutable revision：`c073e143b935a79befd8ab1fd7123796792efad8`；
- archive SHA256：`c3619a177f2b3bd958295bf1fb09653333eb6a735004713158a457a45534a45e`；
- source run commit：`a0001cbc4d4c3e2584ae3bbcff217ce664e5e064`；
- run contract SHA256：`ed1791e148a18f39f3fc04fd0b4aacf66bfc3225e8fc92a38c9c97835e1c61f1`；
- pre-registered golden SHA256：`9bed22b2a378fe35fa1e67ed5edb6f8035f62590c63791c31cbb4336e1067734`；
- 45-record canonical classification SHA256：
  `ad96172b9a3e38098926e58018dc987051a6a98be871b240c93d4281278692c5`；
- formal result SHA256：`78272ee54c1234244239eff080f03d7d97b5973fa5d5cf26fddd77b95794a6e0`。

replay 直接从先验哈希过的 archive bytes 解析 96 个 tar member，不按路径二次打开、不展开 raw trace；
每个 accepted payload 都重新做 strict serialization round-trip 和 AndroidWorld bridge construction。结果只保存
raw output SHA，不保存 native output text。整个 replay 没有 import/load model，没有 policy/teacher forward、
generation、retry、top-up 或 confirm access。

## 复现

先 checkout 并 push exact audit commit，取得上述 immutable archive，然后从 `code/` 运行：

```bash
python3 -m scripts.replay_restoration_v2_parser_compat \
  --repository-root .. \
  --audit-git-commit fc3adf13d48bb016014f7efa62bd27c8a4d12f49 \
  --archive /absolute/path/restoration-v2-substrate-screening-20260715T182823Z.tar.gz \
  --output /new/absolute/path/formal_result.json
```

CLI 会从 committed artifact/golden manifests 自动取得 HF identity、archive/run hashes 和 expected
classifications；这些值不能由 argv 自由替换。output 使用 exclusive create，已有文件不会被覆盖。
