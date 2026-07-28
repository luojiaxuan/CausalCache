# Run 记录:MobileWorld closed-loop B 剂量扫描(B∈{1,2,8})

日期:2026-07-27(启动)。目的:正文 budget-axis 摘要 + appendix 完整表。
B=4 为 primary(已有 3v3 数据);本扫描每个 B 单轮(appendix 口径)。

## Campaign 布局

| campaign | 臂 | host | run root(容器内) | server 口径 |
| --- | --- | --- | --- | --- |
| mw-p-b1-v1 | CausalCache-P, B=1 | hyper01 | /bigdata/mw/runs/mw-p-b1-v1 | 两遍 selector, `--selection-budget 1`, 端口 56141/56253 (GPU0/1) |
| mw-p-b2-v1 | CausalCache-P, B=2 | hyper01 | /bigdata/mw/runs/mw-p-b2-v1 | 两遍 selector, `--selection-budget 2`, 端口 56365/56477 (GPU2/3) |
| mw-p-b8-v1 | CausalCache-P, B=8 | hyper00 | /data/mw/runs/mw-p-b8-v1 | 两遍 selector, `--selection-budget 8`, 端口 57141/57253 (GPU0/1) |
| mw-recent-b8-v1 | HGKV+Recent-8 | hyper00 | /data/mw/runs/mw-recent-b8-v1 | adapter-only(runner 侧 recent 选择), 端口 57365/57477 (GPU2/3) |

冻结组件不变:HGKV lora-step300.pt(sha256 5720…67a9)+ two_tower gold-trained
selector(marginal_scorer.pt),beam=3;只改 selection budget 与 runner config
(`causalcache_mobileworld_official_b{1,2,8}_hgkv_sel_v1.json` /
`b8_hgkv_v1.json`,b8 为 b4 的忠实克隆,仅 memory_budget=8、last_image=9、
selection_budget=8 与输出路径不同,见 cc0c74e)。

对照基线:B1/B2 用用户冻结 recent 曲线(31/117, 31/117,commit a5021f3;
adapter-neutrality 已在 B4 双平台验证,写作时披露此近似);B8 recent 用本次
mw-recent-b8-v1 新数据。

## 舰队与容器

- 每 host 16 个模拟器容器(hyper01 前缀 `sglang-omni-jaxan-mws-`,种子
  h01-dose-20260729;hyper00 前缀 `sglang-omni-jaxan-mwh-`,种子
  h00-dose-20260729),每 campaign 分 8 个,4 shard × 2 envs。
- 计算侧仍是每 host 单容器 `sglang-omni-jaxan`;模拟器容器为一次性 fan-out,
  campaign 结束后删除。
- hyper01 首次舰队启动失败(host /tmp 被清,launcher 不在),已重新发货并
  由 host 侧 `/tmp/dose_h01.sh` 等 manifest 自动接管(URL 改写→切分→起
  server→挂 supervisor)。hyper00 对应 `/tmp/dose_h00.sh` 已完成
  (DOSE_H00_LAUNCHED)。

## 存活与恢复(全局规则要求项)

- 心跳:每 shard `heartbeat-shard-N.json`(supervisor 每轮+每 60s 写),
  字段含 phase/completed_tasks/pending/attempt。
- 断点续跑:supervisor 每轮用 make_subset.py --pending-from 反推未完成任务,
  runner 崩溃自动下一轮;连续 3 轮 pending 不降判 STALL 停机报警。
- 监控:本机(容器外)持久 Monitor 每 5 分钟轮询两 host 的
  `/tmp/dose_check.sh`,stale 阈值 20 分钟(ALERT-STALE),另报
  ALERT/STALL/FAILED 与 hyper01 接管完成事件。告警路径:Claude 会话事件。
- 完成判定:每臂 117 任务(shard 30+29+29+29)全部落盘 result 后 reduce;
  报告前核对分母。

## 扩容(2026-07-27,用户指示每机 6 卡加速)

- 模拟器:每机 +16(前缀 mwh2-/mws2-,种子 h00-dose2/h01-dose2-20260729),
  总 32/机;每 campaign 16 个 = 4 shard × 4 envs(NUM_ENVS 2→4)。
- server(每机 6 卡):hyper00 = P-B8 ×4(GPU 0,1,4,5;该臂 GPU 100% 为瓶颈)
  + recent ×2(GPU 2,3);hyper01 = B1 ×3 + B2 ×3(canonical GPU0-3 四个 +
  aux 容器 sglang-omni-jaxan-2(device 4-7,IP 172.17.0.38)两个:56589/56813)。
  canonical 只挂 0-3 卡,跨容器 server 用 supervise_shard_b{1,2}e.sh
  (arg4 = 完整 endpoint URL)。
- 事故记录:扩容脚本 `docker exec` 少 `-i`,merge heredoc 静默空跑 →
  supervisor 以 4 envs 对 2-模拟器 manifest 崩溃循环
  ("fleet has fewer environments than requested"),16 shard 全部烧满
  STALL 上限退出。修复:`docker exec -i` 重跑 merge(每 shard 4 模拟器)、
  supervisor 全量重启,pending-from 断点续跑无数据损失。
  教训(重复第二次):**容器内 heredoc 必须 `docker exec -i`**,且脚本要
  校验关键步骤输出非空,不能只看退出码。

## 暂停(2026-07-27,CUDA 驱动更新)

- 暂停时进度:P-B8 44、recent-B8 76、B1 32、B2 33(/117)。
- 已执行:supervisor/runner/server 全部杀净,GPU 容器 docker stop
  (h00 sglang-omni-jaxan;h01 canonical + sglang-omni-jaxan-2)。
  两机我方 GPU 显存清零(h01 GPU7 剩 10GB 为他人 root 进程,与我们无关)。
- 模拟器容器(64 个,无 GPU)保持运行;若驱动更新伴随重启则全灭,
  恢复脚本会用同种子重启舰队(名字/端口确定性复现,fleet-shard manifest
  无需改动)。
- 恢复:`code/scripts/ops/resume_dose_h00.sh` / `resume_dose_h01.sh`
  (Git 为持久层;hyper01 数据盘另有副本 /data04/jaxan/mw/)。流程:
  docker start → 模拟器检查(<32 则同种子重启舰队)→ 6 卡起 server →
  health-wait → supervisor 重启(pending-from 断点续跑)。h01 aux 容器
  IP 若变化脚本会打印提示(endpoint 数组用 $AUXIP 自动带入)。
- 所有轨迹/心跳/manifest 均在持久数据盘(h00 /data02/jaxan、
  h01 /data04/jaxan),重启无损。

## 状态

- 2026-07-27:四臂扩容后 RUNNING,随后因驱动更新暂停(见上节)。
  结果落盘后更新本节并入 data/results。
