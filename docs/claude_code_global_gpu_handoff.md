# Claude Code 全局规则与 GPU skills 迁移交接

> 交接对象：Claude Code。本文是可直接执行的迁移说明，不是背景讨论。
>
> 当前状态：Codex 侧规则和 skills 已就绪；Claude Code 侧尚未迁移。迁移时必须合并现有配置，不能覆盖。

## 1. 目标与边界

把当前 Codex 的全局协作规则和 GPU 执行能力迁移到 Claude Code，同时保持以下行为不变：

- CPU 工作默认尽可能使用可有效扩展的全部 CPU 核；只在实测吞吐、内存或 I/O 反向恶化时减少 worker；
- GPU preflight 对候选机器并行执行 **5 秒连续 0% utilization** 检查，清理符合条件的 idle Docker container，并返回全部空闲 GPU id；
- rollout、inference、evaluation 和 generation 使用跨机器独立 data-parallel shards，不建立跨机同步 NCCL/DDP；
- 非 Taurus/Aries 机器每个 Claude task 默认最多使用 4 张 GPU；Taurus/Aries 可使用全部有用空闲 GPU；
- 直接启动完整选定 allocation，不强制先做 1--2 GPU smoke、全机 sweep 或 startup utilization gate；失败、无进展、OOM 或明显过慢时再诊断；
- 只在用户明确要求持续监控，或运行已经表现出不稳定时启动 utilization monitor；
- Docker/container 是可丢弃的，重要状态必须进入个人 persistent storage；不做 shared host 全局 prune；
- Git/Hugging Face source-of-truth、默认 reviewer 和现有安全边界保持不变。

本次迁移不应：

- 修改或删除 `/Users/luojiaxuan/.codex/` 下的任何源文件；
- 覆盖 `/Users/luojiaxuan/.claude/CLAUDE.md` 中现有的 no-attribution 规则；
- 在验证阶段真实清理 container、拉起 GPU job、建立/停止 tunnel、上传 HF artifact 或请求 GitHub reviewer；
- 把 Codex 专用的 `agents/openai.yaml` 复制给 Claude Code；
- 改写任何已经 committed 的 frozen experiment contract。

## 2. Claude Code 的原生映射

迁移采用 Claude Code 官方约定：

| Codex 侧 | Claude Code 侧 | 处理方式 |
|---|---|---|
| `~/.codex/AGENTS.md` | `~/.claude/CLAUDE.md` | 合并规则，并转换 skill 引用；不能覆盖已有内容 |
| 仓库 `AGENTS.md` | 仓库根目录 `CLAUDE.md` | `CLAUDE.md` 只写 `@AGENTS.md`，避免双份规则漂移 |
| `~/.codex/skills/<name>/SKILL.md` | `~/.claude/skills/<name>/SKILL.md` | 保留 frontmatter、references 和 scripts，转换 Codex 路径与调用语法 |
| `$skill-name` | `/skill-name` 或自然语言触发 | Claude Code 直接调用使用 `/name`；自动调用依赖 frontmatter `description` |
| `~/.codex/state/...` | `~/.claude/state/...` | Claude 使用独立状态目录，不干扰 Codex 的现存 tunnel/socket |

相关官方依据：

- [Claude Code skills](https://code.claude.com/docs/en/slash-commands)：personal skills 位于 `~/.claude/skills/<name>/SKILL.md`，支持配套 scripts/references、`/skill-name` 调用和 `${CLAUDE_SKILL_DIR}`；
- [Claude Code memory](https://code.claude.com/docs/en/memory)：用户规则位于 `~/.claude/CLAUDE.md`，项目规则可位于仓库 `CLAUDE.md`，并支持 `@AGENTS.md` import；
- [Claude Code configuration debugging](https://code.claude.com/docs/en/debug-your-config)：先用 `/context` 看实际注入内容，再用 `/memory`、`/skills`、`/doctor` 和 `/status` 检查来源与配置；
- [Claude Code features overview](https://code.claude.com/docs/en/features-overview)：always-on 约束放 `CLAUDE.md`，按需工作流放 skills。

## 3. 源与目标布局

### 3.1 规则源

| 用途 | 源文件 | 迁移目标 |
|---|---|---|
| 全局规则 | `/Users/luojiaxuan/.codex/AGENTS.md` | 合并到 `/Users/luojiaxuan/.claude/CLAUDE.md` |
| Claude 已有全局规则 | `/Users/luojiaxuan/.claude/CLAUDE.md` | 原样保留并作为 merge base |
| CausalCache 项目规则 | `/Users/luojiaxuan/Documents/CausalCache/AGENTS.md` | 由仓库根目录 `CLAUDE.md` import |

迁移前快照锚点：

| 文件 | SHA256 |
|---|---|
| Codex global `AGENTS.md` | `e4aa549d94d24b3427b92ac1fb8076d2c7072a9570d05e8ccac1614127dee972` |
| CausalCache `AGENTS.md` | `6a52ed0f88b0ce583608a6e3e683a815a38285188bc64d79397b8fbf735fbbf2` |
| Claude existing `CLAUDE.md` | `f48f8dec9cc41444fc8362aa0184fd3fd9bea268dca5411dd01b18768897c6e2` |

这些 hash 只标识本交接写作时的源版本。若执行迁移时不一致，先审阅 diff，吸收后续更新；不要为了匹配旧 hash 回滚用户的新规则。

### 3.2 skills 的最小依赖闭包

必须迁移以下 8 个 personal skills。前 6 个构成 GPU 执行闭包，后 2 个是全局规则显式依赖。

| Skill | 角色 | 直接依赖或触发关系 |
|---|---|---|
| `gpu-idle-docker-cleanup` | 单机 5 秒 idle cleanup 与 free GPU inventory | 底层 primitive |
| `gpu-fleet-preflight` | 多机并行 capacity discovery | 调用 `gpu-idle-docker-cleanup` |
| `run-gpu-cluster-job` | 按 host profile 启动单机 Docker GPU job | 失败时可用 tunnel skill；按需 monitor |
| `orchestrate-sharded-rollout` | 根据容量、per-host cap 和 shard 数生成/执行跨机计划 | 调用 fleet preflight 和 single-host runner |
| `gpu-utilization-monitor` | 持续监控用户自己的 GPU job | 只在明确要求或已不稳定时使用 |
| `connect-aries-via-taurus` | 修复 Mac → Taurus → Aries SSH relay | 只在 Aries transport failure 时使用 |
| `maintain-research-sot` | 维护 Git/HF source-of-truth 记录 | 由全局 artifact 规则触发 |
| `request-pr-reviewers` | 请求默认 GitHub reviewers | 由全局 PR 规则触发 |

依赖关系：

```text
orchestrate-sharded-rollout
├── gpu-fleet-preflight
│   └── gpu-idle-docker-cleanup
└── run-gpu-cluster-job
    ├── connect-aries-via-taurus  # 仅 transport failure
    └── gpu-utilization-monitor   # 仅显式要求或 instability

global CLAUDE.md
├── maintain-research-sot
└── request-pr-reviewers
```

源目录统一为：

```text
/Users/luojiaxuan/.codex/skills/<skill-name>/
```

目标目录统一为：

```text
/Users/luojiaxuan/.claude/skills/<skill-name>/
```

复制整个 skill 目录中的 `SKILL.md`、`scripts/` 和 `references/`，但排除每个 skill 的 `agents/openai.yaml`。该文件是 Codex UI metadata，不是 Claude Code skill 的组成部分。

目标布局应为：

```text
/Users/luojiaxuan/.claude/
├── CLAUDE.md
├── skills/
│   ├── gpu-idle-docker-cleanup/
│   │   ├── SKILL.md
│   │   └── scripts/cleanup_idle_gpu_containers.sh
│   ├── gpu-fleet-preflight/
│   │   ├── SKILL.md
│   │   └── scripts/preflight_gpu_fleet.py
│   ├── run-gpu-cluster-job/
│   │   ├── SKILL.md
│   │   ├── scripts/launch_cluster_job.py
│   │   └── references/{hosts.json,hyper-h200.md,b200.md,taurus-aries-a6000.md}
│   ├── orchestrate-sharded-rollout/
│   │   ├── SKILL.md
│   │   ├── scripts/plan_rollout_shards.py
│   │   └── references/rollout-manifest.md
│   ├── gpu-utilization-monitor/
│   │   ├── SKILL.md
│   │   └── scripts/monitor_gpu_utilization.sh
│   ├── connect-aries-via-taurus/
│   │   ├── SKILL.md
│   │   └── scripts/connect_aries.sh
│   ├── maintain-research-sot/SKILL.md
│   └── request-pr-reviewers/
│       ├── SKILL.md
│       └── scripts/request_reviewers.sh
└── state/connect-aries-via-taurus/
```

## 4. 全局 `CLAUDE.md` 的合并要求

### 4.1 保留现有 Claude 规则

当前 `/Users/luojiaxuan/.claude/CLAUDE.md` 已包含以下用户偏好，必须保留：

- commit、PR body 和 code comments 不添加 `Co-Authored-By: Claude` 或任何 AI attribution；
- 不添加 `Generated with Claude Code` footer；
- 不覆盖用户已经设置的空 commit attribution。

以该文件为 merge base，把 `/Users/luojiaxuan/.codex/AGENTS.md` 的全局章节合并进去。不要简单用 `cp` 覆盖。

### 4.2 转换 skill 引用

全局规则里的以下 Codex 表达应转换为 Claude Code 可识别表达：

| Codex 原文 | Claude 目标表达 |
|---|---|
| `Invoke $maintain-research-sot ...` | `Invoke /maintain-research-sot ...` |
| `use $request-pr-reviewers` | `use /request-pr-reviewers` |
| `invoke $gpu-fleet-preflight` | `invoke /gpu-fleet-preflight` |
| `invoke $orchestrate-sharded-rollout` | `invoke /orchestrate-sharded-rollout` |
| `invoke $run-gpu-cluster-job` | `invoke /run-gpu-cluster-job` |
| `invoke $connect-aries-via-taurus` | `invoke /connect-aries-via-taurus` |
| `Use $gpu-utilization-monitor ...` | `Use /gpu-utilization-monitor ...` |
| `using ... in $run-gpu-cluster-job` | `using ... in /run-gpu-cluster-job` |
| `per Codex task` | `per Claude Code task` |

可以保留技术名词、规则内容和章节结构；只需要去掉 Codex 专属调用语法。合并后应保持简洁，避免在全局文件中复制 host commands、mutable GPU facts 或整段 skill 手册。
Claude 官方建议 `CLAUDE.md` 控制在约 200 行以内；当前 Codex global 规则为 137 行，现有 Claude 规则仅 5 行，合并后仍可保持在这一范围。本文的迁移细节不要复制进 always-loaded global 文件。

### 4.3 CausalCache 项目入口

在仓库根目录创建 `/Users/luojiaxuan/Documents/CausalCache/CLAUDE.md`：

```markdown
# CausalCache Claude Code instructions

@AGENTS.md
```

项目规则继续只维护在 `AGENTS.md`。Claude Code 通过 import 读取它，这样 Codex 和 Claude 共用同一份项目 source of truth。

## 5. 每个 skill 的 Claude 适配

### 5.1 通用转换

对 8 个目标 skill 逐一执行：

1. 保留 `SKILL.md` frontmatter 的 `name` 和 `description`。description 必须清楚说明触发条件，因为 Claude 会据此决定是否自动调用；
2. 把描述中的执行主体 `Codex` 改为 `Claude Code`，不改变用户授权、安全边界和 operational semantics；
3. 把用户可直接调用的 `$skill-name` 改成 `/skill-name`；
4. skill 自己的脚本统一通过 `${CLAUDE_SKILL_DIR}/scripts/...` 引用，避免写死用户名和安装根目录；
5. 相对 reference 保持相对于当前 skill 根目录，优先使用 `${CLAUDE_SKILL_DIR}/references/...`；
6. 只把 host facts 放在 `run-gpu-cluster-job/references/`，不要重新塞回全局 `CLAUDE.md`；
7. 保留可执行位；不要复制 `agents/openai.yaml`；
8. 除本节明确列出的 Claude path/state 适配、reviewer source-of-truth 修复和 Taurus/Aries cap 修复外，不在迁移中顺手重构已经验证过的脚本逻辑。

### 5.2 必须修复的硬编码

以下不是可选 cleanup，而是 Claude 安装可运行的必要条件。

#### `gpu-idle-docker-cleanup`

把 `SKILL.md` 中的绝对 Codex script path 改为：

```bash
"${CLAUDE_SKILL_DIR}/scripts/cleanup_idle_gpu_containers.sh"
```

保持 5 秒连续采样和 all-container standing authorization 不变。授权仅允许停止/kill 符合采样条件的 container，不允许删除其他用户的文件、volume、image 或 artifact。

#### `gpu-fleet-preflight`

把 `SKILL.md` 中的 script path 改为 `${CLAUDE_SKILL_DIR}/scripts/preflight_gpu_fleet.py`。

`scripts/preflight_gpu_fleet.py` 的 `DEFAULT_PRIMITIVE` 当前指向 `.codex`。改成：

```python
DEFAULT_PRIMITIVE = (
    Path.home()
    / ".claude"
    / "skills"
    / "gpu-idle-docker-cleanup"
    / "scripts"
    / "cleanup_idle_gpu_containers.sh"
)
```

也可继续支持显式 `--primitive`，但默认值必须能在 Claude personal-skill 安装下独立工作。

#### `run-gpu-cluster-job`

把自身 launcher 改为 `${CLAUDE_SKILL_DIR}/scripts/launch_cluster_job.py`，reference paths 改为 `${CLAUDE_SKILL_DIR}/references/...`。将 failure routing 中的 `$connect-aries-via-taurus` 和 `$gpu-utilization-monitor` 改为 Claude skill 名称。

#### `orchestrate-sharded-rollout`

把 planner 改为 `${CLAUDE_SKILL_DIR}/scripts/plan_rollout_shards.py`。跨 skill 调用不要保留 `/Users/luojiaxuan/.codex/...`：

- 语义级调用直接写 `/gpu-fleet-preflight` 和 `/run-gpu-cluster-job`；
- 若 shell 确实需要 sibling script path，可从 `$(dirname "${CLAUDE_SKILL_DIR}")` 解析 sibling skill 根目录；
- 不要复制 fleet/runner 实现到 orchestrator 内部。

当前 source planner 只有一个全局 `--per-host-cap`，会把 Taurus/Aries 也截到 4 张；这与新的全局规则冲突，不能原样迁移。做一个最小、显式的 policy repair：

- 保留 `--per-host-cap`，默认 `4`，只控制非 Taurus/Aries hosts；
- 新增可选 `--taurus-aries-cap N`；未提供时 Taurus/Aries 不设 cap，提供时两者都限制为 `N`；
- host family 判断使用大小写不敏感的 canonical alias；
- manifest 同时记录 `per_host_cap`、nullable `taurus_aries_cap` 和每台选中机器的 effective limit；
- 同步更新 `SKILL.md` 与 `references/rollout-manifest.md`；默认命令不要传 `--taurus-aries-cap`，只有任务明确要求 Taurus/Aries 也限额时才传。

这不是调度重构：logical shard 分配、host order、resume unit 和 output identity 逻辑保持不变。

#### `gpu-utilization-monitor`

把 monitor script 改为 `${CLAUDE_SKILL_DIR}/scripts/monitor_gpu_utilization.sh`。保留“仅明确请求或已不稳定时启用”的上层策略。

#### `connect-aries-via-taurus`

同时修改文档和 script：

```text
~/.codex/state/connect-aries-via-taurus/
→ ~/.claude/state/connect-aries-via-taurus/

aries-${REMOTE_PORT}-codex.sock
→ aries-${REMOTE_PORT}-claude.sock
```

Claude 和 Codex 的 state/socket 必须并存。迁移或验证时不要停止、复用或删除现有 Codex socket。

#### `maintain-research-sot`

该 skill 没有 script；只需转换 Claude/Codex 主体称呼和 `$...` 调用语法。Git/HF source-of-truth 规则不得弱化。

#### `request-pr-reviewers`

把 script path 改为 `${CLAUDE_SKILL_DIR}/scripts/request_reviewers.sh`。这里存在一个必须在迁移时消除的源内冲突：当前 skill 的 description、正文和 script 仍列出 8 人，而新的全局规则只指定以下 3 人：

```text
Hayden727
yxs
zhaochenyang20
```

新的全局规则是目标 source of truth。同步修改 Claude 版本 `SKILL.md` 的 description/list，以及 `scripts/request_reviewers.sh` 的 `DEFAULT_REVIEWERS` 和 help text，使默认值只含这 3 人；仍保留显式 `--reviewer` 覆盖和“不移除已有 reviewers”的行为。迁移验证禁止真实发起 reviewer request。

### 5.3 不能改变的调度语义

迁移后的 fleet → planner → runner 链路必须保持：

1. 对候选 hosts 并行 preflight；
2. 每台机器做 5 秒连续 0% GPU-utilization sample，符合条件才 cleanup；
3. 返回所有 free GPU ids，不在 preflight 阶段提前截断；
4. planner 再应用 per-host cap，并生成稳定 logical shard ids；
5. 所有选中 GPU 立即启动独立 shards；失败 shard 可恢复/重排；
6. 不建立跨集群同步 collective；
7. launch 后不默认开启持续 monitor；
8. 失败、退出、无进展、OOM 或明显慢时才进行针对性 GPU visibility、Docker、mount、disk、batching、CPU、I/O 诊断。

示例：若 capacity 为 `hyper00=4`、`hyper01=6`、`b200=2`、`Taurus=0`、`Aries=0`，并显式设置每台（包括 Taurus/Aries）最多 4 张，则计划应使用：

```text
hyper00: 4 workers
hyper01: 4 workers
b200:    2 workers
total:  10 workers
```

未选中的 hyper01 GPU 和尚未完成的数据 shards 必须保持可重排、可恢复。

## 6. 推荐执行顺序

Claude，执行本交接时按以下顺序进行：

1. 读取当前源和目标，检查 `git status`、三个规则文件及 8 个 source skill 目录；
2. 为 `/Users/luojiaxuan/.claude/CLAUDE.md` 建立可恢复备份，并记录修改前 SHA256；
3. 在临时 staging directory 复制 8 个 skills，排除 `agents/openai.yaml`；
4. 在 staging 内完成第 5 节的语法、路径和 state 适配；
5. 静态验证 staging；通过后再安装到 `/Users/luojiaxuan/.claude/skills/`；
6. merge 全局 `CLAUDE.md`，保留 existing no-attribution rules；
7. 在 CausalCache 根目录创建只 import `@AGENTS.md` 的 `CLAUDE.md`；
8. 执行第 7 节的无副作用验证；
9. 展示精确 diff、测试结果和仍未执行的 live checks；
10. 仅把 CausalCache 仓库内的 `CLAUDE.md` 和必要 docs 纳入项目 Git，commit 并 push canonical `main`。用户目录下的 personal settings/skills 不进入项目 Git。

若目标 skill 目录已经存在，不要先删除：先 diff，保留 Claude 侧较新的用户修改，再做三方合并。

## 7. 验证协议

### 7.1 文件与语法验证

至少完成：

```bash
# 确认 Claude personal skills 都能被发现
find /Users/luojiaxuan/.claude/skills -maxdepth 2 -name SKILL.md -print

# shell 静态语法；对迁移的全部 .sh 执行
bash -n /absolute/path/to/script.sh

# Python 仅编译 AST，不执行 cluster action
python -m py_compile /absolute/path/to/script.py

# 检查残留 Codex-only 路径、状态目录和调用语法
rg -n '/Users/luojiaxuan/\.codex|~/\.codex|\.codex/state|-codex\.sock|\bCodex\b|\$(gpu-|orchestrate-|run-|connect-|maintain-|request-)' \
  /Users/luojiaxuan/.claude/CLAUDE.md \
  /Users/luojiaxuan/.claude/skills

# 不应迁入 Codex UI metadata
find /Users/luojiaxuan/.claude/skills -path '*/agents/openai.yaml' -print
```

残留扫描预期为空。若 reference 中必须说明 Codex/Claude 共存或迁移历史，可以保留，但需要逐条解释，不能是可执行路径。

`py_compile` 可能生成 `__pycache__`；验证后只清理由此次命令新生成的 cache，不得清理更大目录。

### 7.2 Claude Code 加载验证

在一个新的 Claude Code session 中：

1. 运行 `/context`：确认实际 context 中有 user-level `~/.claude/CLAUDE.md`、CausalCache 根目录 `CLAUDE.md`、imported `AGENTS.md` 和 personal skill descriptions；
2. 运行 `/memory`：确认 user/project memory 文件位置符合预期；
3. 运行 `/skills`：确认 8 个 skills 全部出现；
4. 运行 `/doctor`：排查 skill/frontmatter/config 错误；
5. 运行 `/status`：确认实际用户、cwd 和配置来源正确；
6. 分别用 `/skill-name` 请求“只解释/只渲染命令，不执行”，确认 references 与 scripts 可解析。

### 7.3 无 live mutation 的功能验证

迁移验证只允许 read-only/render-only 路径：

- cleanup：允许 `--help`、参数解析或 mock fixture，不 SSH、不 kill container；
- fleet preflight：允许 `--help` 或注入 mock primitive，不执行真实 `prepare`；
- runner：允许 inspect/render 模式，不加 `--launch`；
- rollout planner：使用 synthetic availability JSON 验证 4/4/2 → 10 workers，不启动 runner；
- monitor：允许 `--help`/语法检查，不连接 shared host；
- Aries tunnel：只做 `bash -n` 和静态路径检查，不运行 `status`、`start` 或 `stop`。当前 source 的 `status` 会在 nested socket 缺失时创建 SSH control master，并非 read-only；
- SoT：只验证 instruction loading，不上传 HF、不改 remote；
- reviewers：只运行 `bash -n`、`--help` 和静态内容断言，不调用不带 `--help` 的脚本路径，更不调用 GitHub write API。确认 `SKILL.md`、`DEFAULT_REVIEWERS` 和 help text 都只包含 `Hayden727`、`yxs`、`zhaochenyang20`，旧的 `Ratish1`、`Ccyest`、`JingwenGu0829`、`JiaxinD`、`SandyLuXY` 均不存在。

第一次真实的 5 秒 cleanup/preflight 应等到一个真实 GPU 任务需要资源时再运行。那时按正常授权执行，不额外增加 smoke gate。

planner 的最小 mock 可以写入临时目录；它不是正式 artifact：

```json
{
  "observed_at": "2026-01-01T00:00:00+00:00",
  "hosts": {
    "hyper00": {"status": "ok", "free_gpu_ids": [0, 1, 2, 3]},
    "hyper01": {"status": "ok", "free_gpu_ids": [0, 1, 2, 3, 4, 5]},
    "b200": {"status": "ok", "free_gpu_ids": [0, 1]},
    "taurus": {"status": "ok", "free_gpu_ids": []},
    "aries": {"status": "ok", "free_gpu_ids": []}
  }
}
```

用 Claude 版本 planner 的 `${CLAUDE_SKILL_DIR}` 实际绝对路径运行：

```bash
python3 /absolute/path/to/orchestrate-sharded-rollout/scripts/plan_rollout_shards.py \
  --inventory /absolute/path/to/mock-inventory.json \
  --total-shards 100 \
  --per-host-cap 4 \
  --taurus-aries-cap 4 \
  --run-id claude-migration-mock \
  --output /absolute/path/to/mock-plan.json

jq '{worker_count, hosts: (.hosts | with_entries(.value = .value.gpu_ids))}' \
  /absolute/path/to/mock-plan.json
```

预期 `worker_count=10`，host GPU 数分别为 4、4、2，且 100 个 logical shard ids 恰好各出现一次。

再加一个 Taurus/Aries 非空 mock：令 `hyper00=6`、`taurus=6`、`aries=5`。不传 `--taurus-aries-cap` 时必须得到 `4/6/5`；传 `--taurus-aries-cap 4` 时必须得到 `4/4/4`。这两条同时验证全局默认与任务级统一限额，不得只测试 Taurus/Aries 为空的情况。

## 8. 完成条件

只有同时满足以下条件，才算迁移完成：

- Claude global `CLAUDE.md` 同时包含原 no-attribution rules 和新的全局 collaboration/SoT/compute/storage rules；
- CausalCache 的 Claude project entry 通过 `@AGENTS.md` 使用共同项目规则；
- 8 个 skills 均被 `/skills` 发现，scripts/references 完整且无 `agents/openai.yaml`；
- 无可执行 `.codex` absolute path、Codex state path 或 `$skill-name` 残留；
- Claude 版 reviewer skill 的 description、正文、script default 和 help 均且仅列出 `Hayden727`、`yxs`、`zhaochenyang20`；
- shell/Python 静态检查通过；
- mock rollout 对 4/6/2/0/0 capacity 与显式全机 cap=4 产出 4/4/2、总计 10 workers；对 6/6/5 的 Hyper/Taurus/Aries capacity，默认产出 4/6/5，显式 Taurus/Aries cap=4 后产出 4/4/4；
- 验证过程没有真实 cluster mutation；
- Claude 报告所有文件 diff、验证命令、结果，以及尚未进行的第一次 live preflight。

## 9. 回滚

如果加载或测试失败：

1. 用备份恢复迁移前的 `/Users/luojiaxuan/.claude/CLAUDE.md`；
2. 只移动或删除本次新建且已精确确认的 Claude skill directories；
3. 不触碰 `/Users/luojiaxuan/.codex/`、Codex state/socket、任何 shared-host container、volume、image、dataset 或 checkpoint；
4. CausalCache 的项目 `CLAUDE.md` 若仅含 `@AGENTS.md`，可保留；若需要回滚，只回滚该新增文件，不修改 `AGENTS.md`；
5. 记录失败的 command、stderr、目标文件 SHA 和最小复现，再修复 staging 后重装。

## 10. 可直接交给 Claude 的启动提示

```text
请完整阅读
/Users/luojiaxuan/Documents/CausalCache/docs/claude_code_global_gpu_handoff.md，
并严格按“推荐执行顺序”完成 Codex global AGENTS.md 与 8 个 personal skills 向
Claude Code 的迁移。

先检查当前源和目标状态；merge 而不是覆盖现有 ~/.claude/CLAUDE.md；保留现有
no-attribution rules；不要修改 ~/.codex 源；不要复制 agents/openai.yaml。将 skill
自身脚本改为 ${CLAUDE_SKILL_DIR} 路径，将跨 skill 调用改为 Claude 的 /skill-name，
并让 Claude 与 Codex 使用不同的 tunnel state/socket。

验证阶段只做静态、mock、render-only 和 Claude /context、/memory、/skills、/doctor、/status
检查，不真实清理 container、不启动 GPU job、不 start/stop tunnel、不上传 HF、
不请求 reviewer。完成后给我精确 diff、测试结果、未执行的 live checks 和回滚点。
```
