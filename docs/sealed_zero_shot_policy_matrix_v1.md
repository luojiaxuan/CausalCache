# Sealed 零样本 policy 评测矩阵 v1(AndroidWorld,预注册)

状态:`FROZEN_BEFORE_EXECUTION`(2026-07-24)。目的:执行
[history-gated mainline v1 合同](history_gated_mainline_v1.md)第 15 节第 12 步——
sealed 零样本 policy 评测(AW 侧)。本文冻结名单、policy、臂与执行协议;发射后不得修改。
不许按 benchmark 结果改 adapter、换 checkpoint 或调协议(canary 只抓工程故障的纪律继续适用)。

## Sealed 名单(roster)

- plan:`data/manifests/androidworld_test_plan_v1.json`,`split=="test"`;
- 文件 SHA256:`091bc49cad174df2df5776da89d044256415d7dd748d13f349b2148ef2e9e977`;
- `instance_records_sha256`:`aaaacb35ddd5533a3736c608ef08842fe72e1734ee7b37a87ecb7fe2933832b2`;
- 构成:冻结 partition(registry SHA `185ae2019706693bd32ecc25ffd0c8f87be87331cae6f7d7e31c91674c962b89`)
  test buckets 8--9 的 25 个模板 × 3 实例(task_index 0/1/2,suite seed 161803)= 75 实例,
  step budget 10--120(官方 `int(10*complexity)`,不得截断);
- 生成:`scripts.build_androidworld_validation_plan --split test --allow-sealed-split`
  (sealed 拒绝保持默认,仅显式开关放行),在 hyper01 pinned AndroidWorld server 环境
  (image ID `sha256:e46fa6709e8c45c3ac091bc43378c27d5021d1c9236da3f674503dc499c0bb28`)内实例化;确定性核验:同环境重建 validation split 逐字节
  复现冻结 SHA `8204e7f832f1...`,test plan 连续两次生成文件级逐字节一致;
- 本名单为正式结果唯一来源;此前任何 dev/canary/ceiling 路径均未触碰 test 模板。

## Policy(3,checkpoint 全部先于本文冻结,只按 Odyssey heldout 选定)

| Policy | adapter_type | checkpoint | rank/alpha |
|---|---|---|---|
| Frozen | 无 | GUI-Owl-1.5-8B(snapshot `code/configs/gui_owl_1_5_8b_snapshot.json`),无 adapter | — |
| Full-layer LoRA | `full_policy_lora` | hyper00 `/data02/jaxan/runs/ody-margin-v2/lora-step75.pt`(s75),SHA256 `75670fa5bb0a492f5ead26212de49b7e1fe8ddf7c25f27a2a2a82a934f14fdc3` | 16/32(全 LM 层 q/k/v/o_proj,vision 冻结) |
| History-gated | `history_gated_kv` | `hg-s100.pt` = hyper00 `/data02/jaxan/runs/hgkv-formal-v1/lora-step100.pt`,SHA256 `8f2cc49e1aa0b06ce231eb54937d813317f5274a799c97b09be7fdb22be46317`(副本 hyper01 `/data02/jaxan/runs/hgkv-eval/hg-s100.pt`) | 8/16(末 8 层 k/v_proj,mask 门控) |

s75/s100 分别由 Odyssey margin-v2 探针与 [正式 gate v1](../data/results/hgkv_gate_v1/README.md)
在 Odyssey tune/heldout 上选定;本矩阵结果不得反向影响任何选择。

## 臂(3)

`summary_B0` / `recent_B4` / `recent_B8`。对应合同 {B0, Recent-B, Full history};B8 为
32k 上下文下确定性可行上限(上限实验 v1 同定义),充当 full-history 臂。

## 执行协议(与 memory ceiling 协议同常数)

- 引擎:`run_androidworld_memory_ceiling_worker.py`,必须显式传 `--allow-sealed-split`
  (worker 新增开关;`_load_plan_instance` 默认拒绝 sealed split 不变,逐局 summary 记录
  `allow_sealed_split`);
- shared-early-decisions = 2;
- parse-retries = 1:主解码贪心,parse 失败后一次温度采样救援
  (`GUIOwlV21GreedyWithSampledRetryRuntime`,temperature 0.7 / top-p 0.95,assay-v2 协议);
- 每图 2,560 effective visual tokens(`EFFECTIVE_VISUAL_TOKENS_PER_IMAGE`);
- 每 cell 每实例单次运行,无重跑;唯一例外:`infrastructure_failure=True` 的 episode 允许
  重试一次,且必须在结果记录中标注(原始失败 json 保留);
- 成功判定:`score==1.0 AND policy_terminated`;setup/infra、parse、executor、terminal
  failure 分开计数,不从 denominator 删除。

## 规模与聚合

- 总量:25 模板 × 3 实例 × 3 policy × 3 臂 = **675 episodes**(每 policy×臂 cell 75 局);
- 聚合:以模板为配对单位(同模板同实例跨 policy/臂配对),模板级配对差 +
  bootstrap CI(10k percentile,与既有化验尺一致);主对照为各 policy 内
  臂间差与固定臂下 policy 间差。

## 执行资源

hyper00 / hyper01 / H100;emulator 与 worker 1:1,单 GPU 单冻结 runtime,episode 间
`empty_cache`;逐局原子写 checkpoint json,`--resume` 语义与上限实验一致。

## 修订(2026-07-24,用户裁定):无效局不消耗重试额度

- **无效局(void)定义**:`infrastructure_failure=True` 且 `model_step_count=0`(环境未初始化、
  策略未行动)的记录属环境事故,**不计为该 cell 的第一次尝试,不消耗 infra 重试额度**;
  隔离存证,不进任何分母。
- "infra 失败可重试一次"仅适用于真实开跑后遭遇基础设施故障的局。
- 实例:worker n5 绑定的 emulator(hyper01 28206)环境僵死,53 局全部 2.5 秒快速失败——
  记录隔离于 `runs/sealed-matrix-v1/full-infra-void-28206/`(hyper01);其重跑(hyper00)为
  正式第一次尝试,自身仍保有一次 infra 重试权。

---

# 修订 V2(2026-07-24):全量 116-template 零样本,删除 margin-SFT 线

## 决策(用户裁定)

1. **paper 全线零样本**:三个 policy —— Frozen / Full-layer LoRA(ody-margin-v2 s75)/
   History-gated(hg-s100)—— 对 AndroidWorld 全部零样本(均在 GUI-Odyssey 训练/选择,
   AndroidWorld 不参与训练与模型选择)。
2. **margin-SFT(v3-e1)整条删除**,不进 paper 任何表(它在 AndroidWorld 上训练过,与全零样本
   叙事冲突;full-layer LoRA 已提供零样本的"全参适配"对照)。
3. **主表 = 全量 116-template 零样本**,每 template 1 实例(task_index 0)。理由:零样本使
   train/dev/test 切分失去意义(切分只为防"在测试集训练/调参",而此处 AndroidWorld 对所有
   policy 都不参与);报全量既是 AndroidWorld 标准做法,也消除"为何只报 25 个 / 是否挑过"的
   质疑;116 个独立 template 的 template 级配对 CI 强于 25×3 聚簇实例。

## 全量主表设计

- roster:`data/manifests/androidworld_full_suite_plan_v1.json`(split="full",116 template ×
  task_index 0,由 test-25 + train-60 + val-31 三 plan 合并,每实例记 origin_split);
- 矩阵:{Frozen, Full-layer(fl-s75), History-gated(hg-s100)} × {summary_B0, recent_B4,
  recent_B8} × 116 template = **1,044 局**;
- 协议其余不变(shared-early 2、parse-retries 1、2560 visual token/图、void 不占额度、
  配对 template 级 bootstrap CI);
- 复用:原 25×3 sealed 子集的 test-25 index-0 局直接进全量表(resume-skip);delta = 91 个
  train/val template × 1 × 3 policy × 3 arm = 819 局;
- 原 25×3 的 index 1/2 局保留为**附录:within-template 实例方差鲁棒性检查**,非 headline。

## split 闸门

`_load_plan_instance` 扩展:split="full" 在 `--allow-sealed-split` 下可载(与 test 同门,
全零样本套件是有意的 opt-in);默认仍拒未知 split。
