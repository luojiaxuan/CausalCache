# Set-utility throughput pilot v2：formal selection NO-GO

## 结论

candidate-schedule key repair v2 已完成唯一 formal four-H200 attempt，并成功写出 metric-only aggregate；repair
interface 本身闭合。预注册 microbatch selector 正式返回 `NO_GO`：12 个 pair 中 8 个完整成功，3 个在 mb1 的
两次 reference generation 间发生 action mismatch，另 1 个在 mb1/mb2 间发生 cross-variant action mismatch；
实际完成 `68/84` native calls，retry=`0`。

observed full-call reserved peak 为 `45,770,342,400` bytes（45.77 decimal GB，占每个 worker 可见显存
30.49%），未触发 80% memory bound。但 all-12 action-equality prerequisite 未满足，因此不能比较完整 teacher-wall
ratio，也没有选择 mb1 或 mb2。正式状态为
`VALID_COMPLETED_SET_UTILITY_TRAIN_ONLY_THROUGHPUT_PILOT_V2_SELECTION_NO_GO_REFERENCE_ACTION_INSTABILITY`。
这是 reference-action stability/selection contract 的 `NO_GO`，不是 processor key repair failure、OOM、
restoration scientific negative 或 task-success 结论。

## Evidence boundary

- source A：`d5e0cca5c5e05d4aeeef74a1bbfae5685a4254c9`；
- execution-envelope B：`e5002d8820b3e4c8c89343b73ba4a5d13aa117c3`；
- execution envelope SHA256：`dd0e64fb40bd39f839a1df91240a1e9a1a528a0bbfe9131d49ba726e8ecf4e3a`；
- aggregate SHA256：`35e0c250232efbd2b9bdccfb1b04a7c372fbed892635342cce4f9f81097ff33f`；
- exact aggregate：[`aggregate.json`](aggregate.json)；完整 Git-safe run record：[`summary.json`](summary.json)；
- 4/4 attempts、4/4 terminals、12 attempted pairs、8 completed pairs、42 generation calls、26 teacher calls、
  34 teacher examples；
- failure classes：`MICROBATCH_1__REFERENCE_ACTION_MISMATCH=3`、
  `CROSS_VARIANT_REFERENCE_ACTION_MISMATCH=1`；
- Hyper00 raw metric-only attempts、terminals 和 logs 保留于
  `/data/runs/causalcache-throughput-pilot-v2-key-repair-d5e0cca`，状态为 `LOCAL_FORENSIC_NOT_UPLOADABLE`；
- 本次未创建或修改 HF artifact；restoration labels、predictor training、matched-NLL 与 closed-loop 全部继续
  locked。

8 个成功 pair 上 mb2/mb1 teacher-wall ratio=`0.997107`，只作 incomplete-subset 描述，不能参与 formal
selection。运行期 10 秒 container-level window 平均利用率为 84%，另有四张 allocation 同时瞬时 100% 的
active-run snapshot；后续逐卡 0% 采样发生在全部 worker 结束后，只表示 post-completion idle。以上 operational
monitoring 不参与 scientific verdict，也不声称每张卡持续达到 80%。

v2 identity 不得重跑、补 state、放宽 equality 或用 partial latency 选择 microbatch。下一步先另立 versioned
train-only action-stability diagnostic，区分 repeated encoding/input mutation 与 BF16/attention-backend 数值不稳定；
formal labels 与训练在该诊断和后续新 throughput identity 通过前继续 locked。
