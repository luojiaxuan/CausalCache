# OSWorld-30 独立复现交叉验证 (2026-07-29)

对 GPT 报告的 OSWorld-Verified 30 步扩展 horizon 结果做四机独立复现。
基础设施:h00(recent shards 0-7,12-15/16)+ h01(recent 8-11)+ h100(sel 0-15/24)
+ Aries(sel 16-23/24);serve_osworld_official_policy.py,--max-steps 30,B=4。

## 结论(n: recent 361/361,sel 357/361)

| 臂 | succ@30 | @15 前缀反事实 | Δhorizon | GPT 对应 |
|---|---|---|---|---|
| Recent-4 | 38.8% | 29.6% | +9.1 | 42.4 / 33.0 / +9.5 ✓ 复现 |
| frozen+selector(无 adapter) | 35.6% | 28.0% | +7.6 | —(GPT 跑的是 HGKV+selector:46.7/+13.4)|

- 配对(n=357):frozen+selector 比 recent **低 3.1pp**。
- **臂口径差异**:本复现 sel 服务器 `adapter: null`(纯 selector),GPT 的 46.7 为
  HGKV+selector。两组数据合并的解释:桌面 30 步下增益需要 adapter 在部署端参与
  (纯 selector 甚至略负),与 mobile 端 HGKV+sel 36.8 > frozen+sel 33.9 同向且更极端。
- 绝对水平系统性低 ~3.6pp:@15 列为前缀反事实(低估)+ 单跑方差。
- 论文中 \method 的 46.7/33.3 依赖 GPT 原始记录,待到货核对;如需完全独立验证,
  需带 HGKV adapter 重跑 sel 臂(~3-4h)。

## 运维教训
- VM zip 多 worker 并发下载会互相追加写坏(h100 曾写出 167GB 损坏包);必须单进程预下。
- `/tmp/docker_port_allocation.lck` 陈锁会饿死后续 worker(h100 shard-4),清锁即愈。
- Aries HF 下载中断可留下同尺寸损坏 safetensors,快照 SHA 校验戳拦截了静默权重漂移。
