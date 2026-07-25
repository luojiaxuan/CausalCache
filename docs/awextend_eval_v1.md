# AW-Extend 冻结 GUI-Owl 评测(B=4)执行记录 v1

状态:**进行中**(2026-07-25 起)。本文件记录 benchmark 来源、我们对环境做的最小改动、
以及三个会影响"能不能和已发表数字直接比"的坑。数据与结果仍为 `PENDING_HF_UPLOAD`。

## 1. Benchmark 来源

AW-Extend 出自 AgentProg(["AgentProg: Empowering Long-Horizon GUI Agents with
Program-Guided Context Management"](https://arxiv.org/pdf/2512.10371),
仓库 [MobileLLM/AgentProg](https://github.com/MobileLLM/AgentProg))。
19 个长程模板,分 compositional 与 iterative 两类,任务名单在其
`eval/android_world/android_world/tasks_long.txt`,通过 `LONG_TASKS=1` 选中。

19 个模板的 `complexity` 全部为 **100**,而 AndroidWorld 的步数预算是
`int(10 * complexity)`,所以每局预算 **1000 步**(stock 任务多在 14~78 步)。

## 2. 我们的执行口径

- policy:冻结 GUI-Owl-1.5-8B-Instruct(`mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faec`),
  官方单体 wrapper 协议,`last_image=5` 即记忆预算 **B=4**;
- roster:`seed=30`、`n_task_combinations=2`,19 模板 × 2 实例 = **38 局**;
- 统计口径与主表一致:**先在模板内对实例求平均,再对 19 个模板做 macro average**,
  分母是模板数不是 38;
- runner 为 `code/scripts/run_official_androidworld.py`(含 2026-07-25 修好的历史对齐)。

**已知口径瑕疵**:38 个实例只有 **35 个不同 goal**,即有 3 个模板在
`task_index=0/1` 上生成了完全相同的任务(参数空间太小)。这些模板的"2 个实例"
实际是同一局重复,必须按模板报告,不能当作 2 个独立样本。

## 3. 环境改动:纯增量 overlay(而不是换成 AgentProg 的 fork)

AgentProg 的 fork 是 **2024 版** android_world,我们 env 镜像里的是 **2025 版**,后者独有
`env/android_world_controller.py` 的 `NONE` a11y 模式、`env/interface.py` 对 STATUS 动作的
短路、可移植 tmp 路径等。**整体覆盖会让环境退版**,所以我们只做增量:

- 新增包 `task_evals/aw_extend/`,里面用镜像版基类定义这 19 个子类;
- `registry.py` 是唯一被改的既有文件,且是 **insert-only**(diff 中被删除/修改行数 = 0);
- 以 bind-mount 方式挂进 emulator 容器,基础镜像保持不变,stock AndroidWorld 跑法不受影响。

验证(真跑,非静态检查):registry 由 **116 → 135**(恰好 +19);19 个任务全部能
`generate_random_params()` → 实例化 → 取 `goal`(即 `android_server.py` 用的同一条路径);
116 个 stock 任务的 complexity / n_rows / app_names / template / mro **零漂移**。
端到端:真 emulator 里 `Expense/Markor/Calendar/Contacts` 四类任务
`/task/initialize` 全部返回 success。

## 4. 三个影响可比性的坑(重要)

1. **AgentProg 的 fork 会顺手改 stock 任务的 complexity**,例如
   `ExpenseAddMultiple` 6→4、`ExpenseAddMultipleFromGallery` 6→2、
   `SimpleCalendarAddOneEventTomorrow` 3.4→2.6,共 10 个。
   因为预算 = 10×complexity,直接整体套用它的 fork 会让 **stock AndroidWorld 的步数预算
   缩水最多 3 倍**,我们既有的 frozen 主表与官方复现会被静默污染。我们拒绝了这些改动。
   反过来说:**如果某篇工作的 AW-Extend 与 AndroidWorld 数字都出自这个 fork,那它的
   AndroidWorld 部分与上游 benchmark 不是同一个预算口径。**
2. **`MarkorMergeNotesLong` 在 2025 版 android_world 上会无条件得 0 分**:它比对
   `adb shell cat` 的输出时没有去掉 `\r`,空行判断 `not content_split[i]` 永远看到 `"\r"`。
   镜像版自带的 `MarkorMergeNotes` 有这个处理,fork 的 Long 版没有。我们按镜像版补上了。
   这意味着该任务的已发表分数依赖于底下 android_world 的版本。
3. **schema 校验是关掉的**:`MarkorFetchMultipleNotesAndSms` / `MarkorTodoList` 声明的
   schema 覆盖不到它们实际生成的参数(list 值、`calendar_params`、`row_objects`)。
   2025 镜像把 `TaskEval.__init__` 里的 `jsonschema.validate` 注释掉了才没炸,
   2024 fork 是开着的。我们保持原样以维持保真,但这是个隐雷。

## 5. 并发结构

emulator 用软渲染(`-gpu off` / swiftshader),**不吃 GPU**;GPU 只服务 policy。所以:

- 每台 emulator 一个容器(`:env` 镜像把 console/grpc/bind 端口写死,一容器只能一台),
  命名 `sglang-omni-jaxan-<MMDDHHMM>e<NN>`,只映射 5000 到不同主机端口;
- **一张 GPU 一个 8B policy 进程**,进程内多线程,每线程独占一台 emulator,
  `generate` 用共享锁串行化(与逐局串行逐字节等价);
- 本轮 38 台 emulator / 4 张 GPU,emulator 数 ≥ episode 数,每台只跑一局,尾延迟最小;
- 实测每台 emulator 约 4 GB 内存、稳态约 0.5 核,冷启动约 13 分钟。

这批 emulator 属于**一次性 fan-out 容器**,run 结束立即 `docker rm -f`,
并用 `docker ps -a --filter name=sglang-omni-jaxan` 核验无残留。

## 6. 复现命令

```bash
# 1) overlay(纯增量,已验证)
#    /data02/jaxan/staging/awextend/overlay -> 容器内 /android_world

# 2) roster(离线生成,不需要 emulator)
python3 code/scripts/build_androidworld_official_seed_plan.py \
  --templates-from <19 模板参考文件> \
  --output awextend_ceiling_plan_seed30_2comb.json \
  --seed 30 --combinations 2

# 3) emulator 舰队
bash awextend_fleet.sh up 38 41001 && bash awextend_fleet.sh wait 38 41001

# 4) 一 GPU 一进程,按 base-url 分线程
python3 awextend_launch_runner.py \
  --plan awextend_plan_19x2.json \
  --ceiling-plan awextend_ceiling_plan_seed30_2comb.json \
  --urls ready_urls.txt --gpus 0,1,2,3 --last-image 5 \
  --output-root <run dir> --repository-root <repo> \
  --model-dir <GUI-Owl 快照> --ocr-model-dir <OCR>

# 5) 收尾
bash awextend_fleet.sh down
```

---

## 7. 结果(2026-07-25,部分完成)

冻结 GUI-Owl-1.5-8B,`last_image=5`(B=4),seed 30 / 2 combinations。
原始报告:`data/results/awextend_b4/awextend_b4_partial.json`。

| 项 | 值 |
|---|---|
| 完成 | **25 / 38 episodes**,覆盖 **14 / 19 templates** |
| template-macro 成功率 | **0.500**,CI95 [0.25, 0.75](template-cluster bootstrap,仅 14 模板) |
| episode-micro 成功率 | 0.560 |
| 终止原因 | **`policy_terminated` 25/25** |
| infra 故障 | **0** |
| 模型步数 | 均值 88.2,最大 689(预算 1000) |

### 按应用家族

| 家族 | macro | 模板数 |
|---|---|---|
| SimpleCalendar | **1.000** | 4 |
| Contacts | 0.750 | 2 |
| Recipe | 0.500 | 1 |
| Expense | 0.333 | 3 |
| **Markor** | **0.000** | 4 |

### 必须一起读的三条限定

1. **0.500 这个数是偏高的**,而且是系统性偏高。未完成的 13 局恰恰是最重的批量录入任务
   (`ExpenseAddMultiple*`、`RecipeAddMultipleRecipes*`、`ExpenseDeleteMultiple*Long2`、
   `MarkorMergeNotesLong`、`MarkorTodoList`),它们要逐条录入 10-20 个条目。
   **缺的正是最难的**,所以真实的 19 模板 macro 只会更低。不能把 0.500 当作 AW-Extend 的
   完整成绩上报。
2. **25 局全部是 `policy_terminated`,没有一局步数耗尽**(预算 1000,实际均值 88)。
   模型总是自己认为做完了。这与主表测到的"错误终止里约 70% 是过早终止"一致。
3. **Markor 家族 0/4 是真实的模型失败,不是环境问题**。抽查
   `MarkorFetchNoteAndSms`(任务:从短信里取文件名 → 在 Markor 打开 → 把内容发回)
   的动作序列:模型打开短信应用后**没有去读收件箱里那条已有短信**,而是直接新建会话、
   输入字面量 `"FileName"` 发出,然后宣称完成。已核实 `initialize_task` 里确实调用
   `adb_utils.text_emulator(...)` 注入了那条收件短信 —— 环境铺好了,模型没去读。
   这是组合式长程任务的典型失败:第一个依赖环节的信息检索失败,后续全部建立在幻觉上。

### 未完成的模板

`ExpenseAddMultipleLong`、`ExpenseDeleteMultipleLong2`、`ExpenseDeleteMultipleSuperLong2`、
`MarkorMergeNotesLong`、`MarkorTodoList`、`RecipeAddMultipleRecipesLong`、
`RecipeAddMultipleRecipesSuperLong`、`RecipeDeleteMultipleRecipesSuperLong`。

跑了 5.5 小时后仍在进行(实测单局最长已到 689 步);按每步约 40-45 秒、预算 1000 步计,
单局最坏可达十余小时,故按实际完成度收尾并显式记录缺口,而不是无限等待或悄悄改分母。
模拟器舰队已在收尾时全部移除,`docker ps -a --filter name=sglang-omni-jaxan` 核验仅剩
每机一个长驻容器。
