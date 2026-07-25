# sparse-history v5 留出集结果与 alignment gate 判定

日期:2026-07-25。留出集 **494 组 / 176 episodes**,CI 为 **episode-cluster bootstrap**。
语料 `gavinlaw/causalcache-sparse-history-guiodyssey-v5@f2144f45`(后续 revision `b3a3a9e3` 补了
dataset card 与图)。报告文件:`runs/eval-v5/gate_report_full.json`、
`runs/eval-v5-epoch1/gate_report_epoch1.json`。

## 1. scorer 可信性:identity 自检精确通过

零初始化 adapter 下 `adapter_on_recent = adapter_on_sparse = 0.00000`,
且 `SA − RA = +0.03350` **精确等于** `frozen_selection_effect`。这是 config 里预置的
恒等断言,成立到小数点后 5 位。

## 2. 主结果

| ckpt | **SA−RA(主 claim)** | SA−R0 | A_c = SA−S0 | **A_recent = RA−R0** | M_shuf | M_irr | M_dup |
|---|---|---|---|---|---|---|---|
| identity | **+0.0335** | +0.0335 | 0 | 0 | −0.0237 | −0.0208 | −0.0217 |
| s25 | **+0.0284** | +0.0478 | +0.0143 | +0.0194 | −0.0202 | −0.0165 | — |
| s50 | **+0.0109** | +0.0924 | +0.0589 | +0.0815 | −0.0088 | −0.0016 | — |
| s75 | **−0.0012** | +0.1203 | +0.0868 | +0.1215 | −0.0022 | +0.0048 | — |
| epoch1 | **−0.0006** [−0.0037,+0.0025] | +0.1175 | +0.0840 | +0.1181 | −0.0009 | +0.0066 | +0.0108 |

三个结论:

1. **主 claim 单调下滑到零,没有中途峰值**:+0.0335 → +0.0284 → +0.0109 → −0.0012 → −0.0006。
   s25 已经比不训练更差,**早停救不回来**。
2. **`A_recent > A_c` 在每一个 checkpoint 上成立**,差距随训练拉大
   (+0.005 → +0.023 → +0.035 → +0.034)。adapter 对"最近窗口"的放大**强于**对"稀疏选点"的放大。
3. **只报 `SA−R0` 会得出完全相反的结论**:它从 +0.034 涨到 +0.118,看起来是巨大成功。
   是 `RA` 这个对照臂把它归零的。**没有 RA,这份工作会把 +0.118 当成方法有效发表出去。**

## 3. adapter 确实学会了区分对错历史 —— 但那不是瓶颈

| | identity | s25 | s50 | s75 | epoch1 |
|---|---|---|---|---|---|
| `B_shuffled = A_c − D_shuf` | 0 | +0.0017 | +0.0083 | +0.0122 | **+0.0136** |
| `B_irrelevant` | 0 | +0.0042 | +0.0191 | +0.0255 | **+0.0274** |

三个 `B_n` 全为正且单调增大;`M_irr`、`M_dup` 由负转正。**correct 与 negative 是可分的。**

但 **drift 爆表**:epoch1 上 `step_shuffled_drift = +0.070`、`irrelevant = +0.057`、
`duplicate = +0.061`,阈值 `< 0.02`,超标 3 倍。权重 2.0 的 anchor 压不住 rank hinge。

## 4. 判定:**不加 alignment gate**

预注册决策链的机械判定给出 (b)(`M_s = −0.0009` 落在 [−0.01,0]),
**但我不采纳这个结论**,因为 (b) 的前提是"仍有希望,继续到 s100",而数据方向相反:

- (b) 要求 `s0→s25→s50` 单调**改善** —— 实际单调**恶化**;
- (b) 要求 drift 下降约一半 —— 实际从 0 涨到 0.057~0.070;
- `M_i = +0.0066 > 0`,已超出 (b) 的窄带。

margin 落进 [−0.01,0] 是**从 −0.024 升上来**的结果,与主 claim 的崩塌是同一件事的两面,
不构成"继续训练会好转"的证据。

**alignment gate 是为"correct 与 negative 在梯度空间不可分"设计的补救。本轮数据表明它们可分
(B_n 全正、两个 margin 转正),所以 gate 不对症。** 真正的病灶是 adapter 对
"历史图存在"本身的无差别放大,而且对 recent 比对 sparse 更强 —— gate 不改变这一点。

### 更对症的方向(未验证,供讨论)

1. **让 RA 进入损失**。当前 config 的 `objective.excluded_arms` 把 RA 排除在外,
   训练全程**没见过**"最近窗口 + adapter 开"这个对照。模型没有任何压力去区分
   "该放大稀疏历史"与"该放大任何历史"。
2. **直接优化 `SA − RA`** 而不只把它当验收指标。当前优化的是 `SA − R0`(reference 是
   bypass 的最近窗口),而 `SA − R0` 恰恰是那个会被历史放大刷高的量。
3. 若要保留现结构,应先做梯度诊断(cosine `C_n` **与**归一化差异范数
   `R_n = |∇ℓc − ∇ℓn| / (½(|∇ℓc|+|∇ℓn|))`,按层按 K 分层)。因为 correct 与 negative 共享
   同一目标动作,cosine 天然偏高,单看它会误判。

## 5. 一个独立发现:prompt 格式是纯成本

`format_effect = R0 − N0 = −0.0217` [−0.0303, −0.0131],CI 不含零。
**我们的 sparse 单轮 prompt 比官方多轮格式差 0.022 nats**,与 adapter 无关。
`deployment_delta = SA − N0` 一直被这个固定成本拖着(epoch1 上 +0.0959)。

若要部署,应考虑把 sparse 选点搬进官方多轮格式,而不是自造单轮格式。

## 6. 复现命令

```bash
# 4 卡分片填缓存(shard_count>1 时强制 --skip-reduction)
for i in 0 1 2 3; do
  python3 -m scripts.score_sparse_history_arms \
    --repository-root <repo> --config code/configs/causalcache_sparse_history_v1.json \
    --model-dir <GUI-Owl 快照> --dataset-root <sparse-v5-final> \
    --include-identity-baseline \
    --checkpoint s25=<...> --checkpoint s50=<...> --checkpoint s75=<...> \
    --shard-index $i --shard-count 4 --skip-reduction \
    --score-cache out/cache --output out/shard$i.json --device cuda:$i &
done; wait

# 归约:全命中缓存,不碰 GPU
python3 -m scripts.score_sparse_history_arms ... \
  --score-cache out/cache --require-cached --output out/gate_report_full.json
```

分片经逐值验证:40 组上单进程 884s vs 4 分片 320s(2.76×),**568 个分数逐位相同、
253 个报告值全等**。
