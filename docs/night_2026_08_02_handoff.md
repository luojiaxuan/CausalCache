# 夜间自主窗口交接(2026-08-01 夜 → 08-02 晨)

任务:查清 desktop/OSWorld **同域闭环不涨**的原因并解决;做修复 1、2,3 可选;写文档。

## TL;DR

**同域不涨的根因找到了,而且不是"方法没意义"。**

1. **头寸存在**:oracle 比 Recent-2 多拿 **+0.102** 效用,我们的 selector 只捕获 **6.5%**;
   教师标签里 **95.3%** 的状态有值得恢复的帧。仓库原来"recent 已吃满"的解释被数据否定。
2. **部署链路有一个可修的校准缺陷**:两塔中的 readout 塔在部署侧被 mask(它需要挂探针
   才能算),而该塔实测**近似一个常数**(均值 +0.065 / 标准差 0.002)。去掉它使
   cheap 塔单独预测在 20,503 条 dev 边上**无一为正**。常数偏置不改 argmax,
   但让**一切阈值/早停判据失效**。
3. **修复后同域首次出现正向**:同批双臂 **32.69% vs 30.47%,+2.22pp**(CI [−0.83,+5.54],
   p=0.10)。点估计较历史 +1.11pp **翻倍**,赢输比 25:21 → 22:14,
   **writer 域从 0 赢 4 输修复为 3 赢 2 输**。**但未达显著,不可声称已解决。**
4. **单轮口径本来就没有分辨力**:两个 98% 行为相同的臂,任务级翻转 **14.4%**、
   差 3.9pp。**OSWorld 15 步单轮 ±4pp 内不可解释为效应**——原来那个"+1.11pp 不显著"
   既不能证明有效也不能证明无效。

## 交付物

| 文档 | 内容 |
|---|---|
| `data/results/indomain_gap_v1/ROOT_CAUSE.md` | **主文档**。根因链、五条备选解释的排除、三项修复、语料坑 |
| `data/results/indomain_gap_v1/README.md` | τ=0 弃权实验(空转)及其为何仍有价值 |
| `data/results/indomain_twoarm_v1/README.md` | 同批双臂第 1 轮,含域级对照与统计力计算 |
| `data/results/selector_deploy_config_v1/` | 部署配置离线诊断(replay_gain 掉 5 倍) |
| `data/results/selector_calibration_v1/` | 校准崩塌的直接测量(两个 bundle) |
| `docs/indomain_defense_draft.md` | **论文表述草稿**,未写入任何 .tex(投稿版归属待你裁定) |
| `code/runscripts/README.md` | 七条运维教训,今晚新增第六、七条 |

## 代码改动

| 文件 | 改动 |
|---|---|
| `train_selector_v4_marginal.py` | `--eval-only`(加载 bundle 只评测,自动跑 readout 开/关两遍)、`--select-by-deploy-config`(按部署口径早停/选快照)、校准诊断输出 |
| `serve_osworld_official_policy.py` | `--selector-min-marginal τ`(弃权 + recent 补齐)、`--selector-score-offset`(校准截距)、审计字段 `promoted`/`filled_recent`/`k_off_recent` |
| `reduce_indomain_twoarm.py` | 同批双臂归约(总体 + 域级 + 跨轮参考附警示) |
| `merge_aries_repair.py`、`analyze_osworld_host_confound.py` 等 | 前序任务 |

## 三项修复的状态

**修复 1(按部署口径选 checkpoint)——已完成。**
在**全量合并语料**上重训(语料跨两机分片,单机只有 69% 的边、80% 读出覆盖,已合并)。

| | 原 bundle | 新 bundle |
|---|---|---|
| 部署口径 top1-regret | 0.0958 | **0.0918** |
| 部署口径 replay_gain_b2 | +0.0023 | **+0.0066**(首次超过 cheap-only 基线 +0.0051) |

**修复 2(弃权 + 校准截距)——已完成并验证。**
关键发现:**单独上弃权是空转**。τ=0 无截距时 97.4% 的可触发步完全弃权,
该臂在 98% 的步上等同 Recent-4。补上截距 +0.0661 后:

| | 无截距 | 有截距 |
|---|---|---|
| promoted = 0 | 97.4% | **0%** |
| promoted = 1 / 2 | 1.4% / 0.03% | **69.8% / 29.8%** |
| 真正偏离 recent | ~2% | **25.7%** |

**顺序不可颠倒:先校准,弃权才有意义。**

**修复 3(特征增强)——未做,但结论已改变。**
新 bundle 里 readout 塔退化成**近乎纯常数**(均值是标准差的 31 倍),说明
**两塔可加结构下 448 维读出基本没被利用**。所以"加图像 embedding"未必有用——
**问题可能不是特征弱,而是架构无法利用高维特征**。若要做,应连架构一起改
(让候选特征与当前状态做交互,而非可加残差)。**这条我没有擅自启动。**

## 正在跑

**同批双臂第 2 轮**(统计力计算显示需约 2.2 轮才能坐实 +2.22pp)。
输出 `{BASE}/twoarm-r2/out-{selfix,recent}`。跑完执行:

```bash
python3 code/scripts/reduce_indomain_twoarm.py \
  --selfix <r1+r2 的 selfix 收割> --recent <r1+r2 的 recent 收割> \
  --output data/results/indomain_twoarm_v1/twoarm_2rounds.json
```

## 需要你裁定的三件事

1. **投稿版归属**:PDF 出自 `jaxan/trusting-zhukovsky-00d2bf` 分支的 main.tex,
   该分支从未合进 main;我这两天的修改都在 main 分支的 main_v2.tex。
   camera-ready 以哪个为基线?**在这之前我不动 .tex。**
2. **B 的工作点**:oracle 头寸 B=2 是 **+0.102**、B=4 只剩 **+0.037**。
   **B=4 恰是最难展示选择价值的工作点**,与全量表"峰值在 B=2"一致。
   你之前定的"就 B=4"是在这个证据之前,可以重新考虑。
3. **修复 3 要不要做**:按上面的分析,建议连架构一起改,不是单纯加特征。

## 已知风险

- **h00 `/data02` 长期 99%**(现剩约 27G)。第 2 轮还需约 4G。够用但薄,
  我删过一次废弃产物;若逼近 15G 需清理已归档的旧轨迹。
- 同域 +2.22pp **尚未显著**,论文中不可表述为"已修复"。
