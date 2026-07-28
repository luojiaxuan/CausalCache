# AAAI 主图 v1：设计、证据与复现

状态：`COMPLETE`（2026-07-27）。两张图已进入 `paper/main.tex`；Figure 1
位于主稿第 2 页页顶，Figure 2 位于第 6 页。论文仍是 `WORKING`，不是冻结投稿快照。

## 交付物

| 图 | 可编辑源 | 投稿矢量 | 预览 |
|---|---|---|---|
| Figure 1：method overview | `paper/figures/causalcache_overview.svg` | `paper/figures/causalcache_overview.pdf` | `paper/figures/causalcache_overview.png` |
| Figure 2：Cart qualitative case | `paper/figures/causalcache_qualitative_cart.svg` | `paper/figures/causalcache_qualitative_cart.pdf` | `paper/figures/causalcache_qualitative_cart.png` |

两图均为 7 inch 双栏宽度。Figure 1/2 高度分别为 3.89/3.47 inch；图内最小
字号为 9 pt（SVG 中 18 个 viewBox unit，按 1008 unit → 504 pt 换算）。
PNG 为 2016 px 宽，即 288 dpi。颜色同时由边框、虚线、粗线和斜线填充编码，
灰度渲染仍可区分 recent、restored 和 wrong restoration。

生成源为 `paper/figures/build_causalcache_figures.py`。Figure 1 不把事件
“删除/保留”误画成 memory selection：每个事件的 summary 一直存在，promotion
只从 cold visual archive 重新挂接真实截图。Figure 1 同时固定以下语义：

- Recent-4 与 CausalCache 都严格使用 `B=4` 张 active history images；
- `k` 是部署后的 realized replacement count，不是预设参数；
- HGKV 只作用于 restored history-image tokens，其他 token bypass；
- CausalCache-P 是 proposal-conditioned 默认部署路径；
- CausalCache-LA 用虚线表示效率分支，不是默认主方法；
- archive pixels 是检索得到的真实历史像素，不从 summary 生成。

## Figure 2 的真实案例审计

Figure 2 使用 `CartInfoNotificationTask` 的真实闭环轨迹，不是合成故事板。
机器可读审计见
`data/results/mobileworld_hgkv_selected_b4/qualitative_cart_audit.json`。

在 CausalCache-P r2 的第 15 步：

- Recent tail 为 `[11,12,13,14]`；
- selector 真实选中 `[6,7,8,12]`；
- realized `k=3`，因此不能按预设画成“一张升、一张降”；
- event 7 的 archived screenshot 含 `经典白色T恤`、`保湿面霜套装` 和订单号
  `639281475036294`；
- policy 输入并发送
  `经典白色T恤,保湿面霜套装,639281475036294`，evaluator score 为 `1.0`。

独立 Recent-4 r2 在后续第 23 步实际输入
`Women's Cotton Short Sleeve T-shirt, 202306010001`，evaluator score 为
`0.0`。图只把这个 episode 当作机制例证；总体效应仍由正文的配对统计支持。

selector 使用精确 bundle 和该轨迹离线重放。以正确输入、Recent 实际错误输入和
click-input 三种 proposal action 重算，均得到 `[6,7,8,12]`，所以这次 allocation
不是特定措辞偶然触发的图示选择。

原始轨迹属于 `LOCAL_PRIVATE_RAW_TRACE`，不进入 Git：

- selected r2：hyper01
  `/data04/jaxan/mw/runs/mw-hgkv-sel-b4-r2/shard-0/trajectories/CartInfoNotificationTask/`；
- Recent-4 r2：hyper00
  `/data02/jaxan/mw/runs/mw-recent-r2/shard-0/trajectories/CartInfoNotificationTask/`。

图中只提交去除收件人姓名、地址和手机号后的确定性 crop：

- `cart_order_evidence.png`：selected step-7 screenshot 的
  `1080x1150+0+480` crop；
- `cart_message_input.png`：selected step-15 screenshot 的
  `1080x300+0+1980` crop。

## 复现与验证

```bash
make figures
make paper
```

`make figures` 重新生成 SVG，用 `rsvg-convert` 导出 PNG，并先生成 PDF 中间件；
随后 Ghostscript 以 `-dNoOutputFonts` 将 PDF 文字转成矢量轮廓。这样保留 SVG
作为可编辑 canonical source，同时让投稿 PDF 不含 Type 3 字体资源。最终验证包括：

- SVG 可被 XML parser 读取，且没有外链图片依赖；
- standalone PDF 页面尺寸分别为 `504x280.08 pt` 和 `504x249.84 pt`；
- 投稿用 figure PDF 的文字均已转成矢量轮廓，`pdffonts` 不返回任何字体；
  9 页主稿只含 Type 1/TrueType 字体，零 Type 3；
- Ghostscript 完整解析两张图和 9 页主稿；
- 彩色与灰度各自从 PDF 重新 rasterize 后目视检查；
- 主稿为 US Letter、9 页，references 在第 7 页自然开始，第 8--9 页仅
  references；
- LaTeX 日志无 overfull box、undefined reference 或 fatal error。

SVG、PNG 与审计 crop 是稳定内容身份，当前 SHA-256 如下。PDF 是 SVG 的
font-outline 派生物；Ghostscript 可能改变等价 PDF 的对象级序列，因此不把
其二进制 hash 当作 canonical identity。

| 文件 | SHA-256 |
|---|---|
| `causalcache_overview.svg` | `bb2574354f0a20f506c47a529a928bfa287b9054916e69af2dabe6bf08670fd4` |
| `causalcache_overview.png` | `bbfbaaa5d83503cc10f51e1d3473711627b6a90b82a9552406ed9f8b4b6cffa9` |
| `causalcache_qualitative_cart.svg` | `39a4d4e4e014cf3bc46a85dcacdde7a3cbc6f60eca6fd7675d2c027d4102bc7a` |
| `causalcache_qualitative_cart.png` | `fd3f89efba8419c8757dc78ea1288fb37c2a3d79b4a33a892d4eff426ff0147a` |
| `cart_order_evidence.png` | `0dd574a79528e3df8852fb4ef5167b9876e62c9ed28b2f4b7a6a1dd56a1fc20e` |
| `cart_message_input.png` | `36f138cf5ec54a66133b414a9c71992d72abd32ce1e764725621db8bcc2aba53` |
