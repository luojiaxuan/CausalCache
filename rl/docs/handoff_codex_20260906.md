# 交接文档:CausalCache 记忆控制器线(B-pilot)→ Codex session(2026-09-06 22:45 PT)

读者是接手的 agent。本文自足:不需要看之前任何对话。事实以 Git 台账为准,本文只做导航与压缩;冲突时以
`rl/docs/ledger_memory_rl_20260904.md` 最新条目为准。仓库公开:https://github.com/luojiaxuan/CausalCache ,交接时 main = `f157af2`。

## 0. 三句话现状

1. **研究问题**(用户定的):GUI agent 的记忆余量 + "视觉历史不该默认取最近帧"。不做 MementoGUI 式的记忆控制器方法论文。
2. **今天定下的核心发现**(冻结 UI-Venus-2-9B / GUI-Owl-1.5-8B,MobileWorld 内自建的 48 个记忆关键任务,决策步上换上下文):
   老截图的价值只在**任务事实没留在文本历史里**时出现(Mail 两族交互项 +46.7pp,零件图 +80pp);该保留的是**请求轮 + 证据轮**而不是最近两轮
   (GUI-Owl 同 token 43% vs 0%);冲突时数字听文本、形状听图;文本详细时扔掉最近两帧省 46% prompt token 精度不变,但"以图换文"不成立。
3. **用户判定现稿构不成顶会论文;GPT-6 Pro 外审同意并给出路线**(`rl/docs/reviews/top_venue_review_20260906.md`):缺的是**有外部效度的
   可复用方法论**,不是学个控制器;推荐路线 1(benchmark + 干预有效的诊断方法论),**先跑 §3.1 判别实验**(见 §5)。**等用户拍板,本线未发射新实验。**

## 1. 必读文件(按顺序)

| 文件 | 内容 | 读法 |
|---|---|---|
| `rl/docs/ledger_memory_rl_20260904.md` | 唯一现行判断清单。§2 判断表 J1–J15;§5 时间倒序台账(**最新在最上**);§7 犯过的错;§8 待用户裁决 | 先读 §0、§2,再读 §5 从顶到 "2026-09-06 11:10 PT" 条。注意 §5 顶部的时间标签更正说明(09-06 下午标 16:55–18:55 PT 的条目实际快约 1 小时;标签作键不改) |
| `rl/docs/story_20260905.md` | 论文故事线,§2.12–2.13 是今天的结果与口径 | 措辞已被外审要求收敛为条件效应(见 §4 砍列表) |
| `rl/docs/b_pilot_design_20260906.md` | 任务族设计、三档判定(§5)、延后揭示草案(§7,**已被 Pro 外审否定**:文件初始化时就在只是程序上隐藏) | §5 的闸门定义仍有效 |
| `rl/docs/reviews/top_venue_review_20260906.md` | GPT-6 Pro 外审全文 + 采纳表 | 下一步的依据 |
| `rl/docs/reviews/fidelity_axis_review_20260906.md`、`next_steps_review_20260906.md`、`paper_direction_review_20260906.md` | 之前三轮外审(GPT-6 极高档) | 已采纳项都进了台账 |
| `rl/docs/results/deploy2_four_models_20260906.txt` | 部署布局四模型重测表 | 背景 |
| artifact 故事页 https://claude.ai/code/artifact/40dca268-1b07-47b0-8398-c185a7cf4f08 | 给人看的版本 | 与 story 同步 |

## 2. 关键数字(全部可在 hyper00 `/data01/jaxan/rl_v2/pilot/eval_*.jsonl` 复算)

### 2.1 Mail 两族(QuoteRecall 报价回忆 + OrderAddressJoin 订单-地址联接),Venus,30 个 checkpoint,16 对孪生

| 历史文本 | text_only | rec2 | ctrl_at_turn(同龄无证据轮) | src_at_turn(证据轮) | swap→孪生答案 | gold_text | 文件 |
|---|---|---|---|---|---|---|---|
| 整段 think(官方) | 1.000 | 1.000 | 0.967 | 1.000 | 0.143 | 1.000 | eval_venus_mail_text.jsonl |
| think 首句 | 0.867 | 0.900 | 0.867 | 0.933 | 0.214 | 0.767 | eval_venus_mail_oneline.jsonl |
| 只留 action + 请求帧 | 0.000 | 0.000 | 0.000 | **0.533** | 0.357 | 0.933 | eval_venus_mail_action_req.jsonl |
| 只留 action(事实删) | 0.000 | 0.000 | 0.000 | 0.567 | — | 0.933 | eval_venus_mail_fx_action.jsonl |
| action + "Noted: <事实>"(事实留) | 0.933 | — | 0.800 | 0.900 | — | 0.967 | eval_venus_mail_fx_action_fact.jsonl |

交互项 [(src−ctrl)_删 − (src−ctrl)_留] = **+46.7pp,95% CI [+25.0, +66.7]**(按 (族,对) 16 簇聚类)。

### 2.2 GUI-Owl 教师强制(沿 Venus 轨迹逐屏自写一行 conclusion;"受控回放复现",不是自然 agent),30 个 checkpoint

| text_only | rec2 | req_keep | ctrl_req_keep | src_keep | **src_req_keep** | src_req_before(参考轮前置,Noted 变体) | gold_text |
|---|---|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0.000 | 0.200 | **0.433**(9.1k tok,与 rec2 同预算) | 0.467(14.3k tok) | 1.000 |

文件:eval_owl_tf_mail.jsonl / eval_owl_tf2_mail.jsonl / eval_owl_tf_before.jsonl;前缀 rl_v2/pilot/prefix_owl_tf/。

### 2.3 零件图配对 PartMatch v4(四候选,Documents/Candidates,2×2 网格),Venus,15 个 checkpoint,按点击落格判

| 历史文本 | text_only | rec2 | ctrl | src | swap→孪生 | gold | 孪生翻转 | src−ctrl |
|---|---|---|---|---|---|---|---|---|
| 整段 think | 0.867 | 0.933 | 0.933 | 1.000 | 0.800 | 0.867 | 8/11 | +6.7 |
| 只留 action(描述删) | 0.000 | 0.000 | 0.000 | **0.867** | 0.867 | 0.800 | 7/11 | **+86.7 [+60, +100]** |
| action + 自己那句描述 | 0.933 | 0.867 | 0.933 | 1.000 | **0.867** | 0.933 | 9/11 | +6.7 |
| 无文本 | 0.000 | 0.000 | 0.000 | 0.333 | 0.200 | 0.000 | 1/11 | 105 解码里 64 无点击,分布外 |

文件:eval_pm4_venus_{full,notext,action,action_fact}.jsonl;规格 specs_prefix_venus_pm4.jsonl(**用 rl_v2/pilot/task_seeds.txt 的四候选种子**)。
GUI-Owl 自然前缀在四候选版只到达 5 个决策态(不引用);Pictures 六候选版 n=16:src_keep 0.375 vs ctrl 0.125(eval_owl_pm2.jsonl)。

### 2.4 预算(Venus,Mail 30):整段文本+最近两帧 11,160 tok 答 100%;整段文本不给老图 6,048 答 100%(−46%);think 首句 4,542 答 87%;
只留 action + 请求帧 + 证据轮:原尺寸 9,091 答 57%,边长 0.5 → 5,283 答 53%,0.35 → 4,615 答 50%,0.25 → 4,295 答 33%。一张全分辨率截图 ≈ 2,500 tok(两模型同)。

### 2.5 自然状态背景:MobileWorld 117 题自然轨迹 337 状态里只有 1 个需要回看老帧;部署布局四模型重测:最近两帧最高、无关帧拿走一半(格式效应)、
三个 SFT 臂之间无差别(results/deploy2_four_models_20260906.txt)。

## 3. 基础设施(hyper00 = 主场;别的机器没有本线的东西)

- 根目录 `/data01/jaxan/`。脚本都在根下:
  - `pilot_checkpoint_eval.py`:干预矩阵评测器。`--backend owl|venus`,`--text-mode full|oneline|action|action_fact`,`--no-text`,`--scales 0.5,0.35`,
    `--conds a,b,c`(过滤条件),`--rescore`(按已存 decodes 与**规格文件**的 expected 重判)。条件名:text_only / rec2 / src_keep / ctrl_keep / swap_keep /
    src_req_keep / ctrl_req_keep / req_keep / src_before 等(owl);text_only / rec2 / ctrl_at_turn / irr_at_turn / src_at_turn / swap_at_turn / gold_text /
    src_at_turn@<scale>(venus)。每行记 `ptoks`(vLLM prompt_tokens,含图)、`leak{goal,hist}`。
  - `pilot_build_specs.py`:从前缀目录建 checkpoint 规格(决策步、源帧、对照帧、请求帧、孪生、同族无关帧;`--contact` 出拼图供人工核对)。
    **一律用绝对路径 `--prefix-dir /data01/jaxan/...`**(相对路径的规格在别的 cwd 下找不到 traj.json,今晚两条链因此空跑)。
  - `pilot_gates.py <eval.jsonl> [--source ..] [--ctrl ..] [--swap ..]`:三档判定(设计稿 §5),按对聚类 bootstrap。
  - `pilot_owl_tf.py`:GUI-Owl 教师强制前缀生成。`decode_ctx.py`(GUI-Owl 部署布局构造器 messages_deploy / messages_hybrid)、
    `venus_oracle.py`(Venus 官方协议 messages)、`guiowl_oracle.py`(parse_action / extract_conclusion / post)。
  - 链脚本 `pilot_*.sh`(采集/矩阵/接管容器),模板见 `pilot_pm4_venus_all.sh`;都 `source cc_container_lib.sh`。
  - `cc_container_lib.sh`:alloc_name / reg / pick_gpu / pick_port / rm_own。**注意:source 时会重新生成 OWN,外部传入的 OWN 会被覆盖**,
    接管别的脚本起的容器时 rm_own 会 "SKIP (not mine)",需按名 `docker rm -f` + `sed -i "/^<name>\t/d" ~/jiaxuanluo-map.txt`(只删本线自己建的)。
- 数据 `/data01/jaxan/rl_v2/pilot/`:前缀 prefix_{base,venus,owl_tf,venus_pm4,base_pm4,venus_qr2,...}/<Task>/{traj.json,screenshots/,result.txt};
  规格 specs_*.jsonl;结果 eval_*.jsonl/.txt;拼图 contact_*/;种子 task_seeds.txt(当前 = Mail 两族 + 四候选 PartMatch)、task_seeds_v6.txt(六候选旧版)。
- MobileWorld:`/data01/jaxan/mw/MobileWorld`(uv 环境;`uv run mw eval --agent_type ui_venus2|gui_owl_1_5 ...`)。任务类源文件在
  `src/mobile_world/tasks/definitions/work/{quote_recall,order_address_join,part_match}.py`(Git 副本 `rl/cua/mw_tasks/`),**注册表在模拟器容器内的
  /app/service 拷贝里**,改任务后必须 `bash /data01/jaxan/install_pilot_tasks.sh sglang-omni-jaxan-p12 sglang-omni-jaxan-p13`(会重启任务服务,只在模拟器空闲时做)。
- 模拟器:`sglang-omni-jaxan-p12/p13`(端口 6812/6813,CPU 容器,map 写明保留至 2026-09-08 PT)。另一条 session 用 p00–p11、p14–p33,**不要碰**。
- 模型:Venus `/data01/jaxan/models/UI-Venus-2-9b`,GUI-Owl `/data04/jaxan/models/GUI-Owl-1.5-8B-Instruct`;镜像 `vllm/vllm-omni:dev`;
  vLLM 起法见任一链脚本(`--max-model-len 32768 --limit-mm-per-prompt '{"image":8}'`,端口 172.17.0.1:412xx→8000,`-e PYTHONPATH=/pyshim`)。
- 交接时状态:本线 hyper00 上 **无 GPU 容器**、无后台链;账号 GPU 上限 4 张(另一条 session 通常占 2 张)。

## 4. 今天踩过的坑(不要再踩)

1. **端点**:PartMatch 不能按"文本里出现文件名"判命中——Venus 的 think 会把候选文件逐个点名,所有条件同为 0.80;改为只看点击落格(GUI-Owl 看动作行)。
2. **种子过期**:改任务定义后 `task_seeds.txt` 必须从容器日志按时间戳重生成(loguru 时间戳是 UTC,docker `--since` 用宿主本地时 UTC+8,别混用)。
3. **规格相对路径**(见 §3)。
4. **Venus 无文本口径分布外**:一半以上解码没有动作,只作诊断,不引用绝对值。
5. **孪生撞车**:PartMatch 四候选下 A/B 样品需强制不同(part_match.py 已改,v5 未重采)。
6. **Files 视图**:Pictures/Documents 下的图片文件夹是两列网格且切底行(六候选要滚动),四候选才一步可点。
7. **共编台账**:另一条 session 也在改 ledger;每次编辑前 `git show origin/main:<file> > 本地` 再套增量,plumbing 提交时父为最新 origin/main;
   09-06 因暂存副本过期覆盖过对方的行(§7 有记录)。
8. **共享主机**:容器名 `sglang-omni-jaxan-<n>`、`--init`、只暴露分配的 GPU、创建即登记 `~/jiaxuanluo-map.txt`、结束当场删;绝不 rm 别人的容器。
9. **时间**:给用户看的时间一律 PT 并核对换算(今天下午的台账标签快了约 1 小时,已加说明)。

## 5. 下一步(等用户拍板;GPT-6 Pro 外审 §3.1,本 session 未发射)

**判别实验**(0.5–2 GPU-天):80 个**新**场景(不用现有 48 个,它们只作开发集)、两个 actor、两种请求时机(一开始就知道 / 浏览与记忆写入之后才
生成——todo.txt 方案不算,须运行时注入或 ask_user)、**称职的在线结构化笔记 + caption/OCR 检索**作文本基线、历史预算 2k/4k/8k/16k 扫描。
判据:若文本记忆在现实预算内全程占优 → 放弃"图像控制器"论点,本线收成诊断结果;否则进入路线 1 全量(500 场景、≥20 留出模板、≥6 app、4 actor、
AndroTMem 200 条 + MemGUI 全 128 题闭环,12–30 GPU-天 + MemGUI 判分预算)。功效:检出 10pp 需 160–315 个独立场景。

外审要求砍掉的措辞(改稿时执行):联合 RL;"训练布局决定收益"的标题;"recency 有害 / 形状听图数字听文本"当普适定律(改条件效应);oracle 上界当成果;
"46% 效率且精度不变"(需全流程成本 + 非劣性);recent-2 当官方基线(GUI-Owl 官方 MobileWorld 默认不带历史图)。

## 6. 待用户裁决(台账 §8 也有)

- 是否接受路线 1 并先投 §3.1 判别实验;通过后是否批全量(GPU-天 + MemGUI 判分预算,每臂约 $10–15)。
- MemGUI 上 Venus 三臂(需先把 ui_venus2 agent 移植进 MemGUI 评测框架;后端模拟器 mga_* 只剩 2 台健康)。
- 公开仓库的 git 历史里仍有合作者邮箱(未重写历史);冗余镜像 repo `luojiaxuan/CausalCache-review` 未删。

## 7. 规则摘要(来自用户全局规则,操作层面最常用的)

- 外审:方向/结论不确定、发射大实验前、替用户做默认决定前,用网页 ChatGPT 临时会话;默认最新旗舰的次高档(极高),顶配 Pro 只在用户点名时;
  提示词纯英文单行 ≤1500 字符 + FORMAT 段 + 给公开仓库 `blob/<sha>/<path>` 链接;先写审前判断;全文落 `rl/docs/reviews/`;区分同意/改判/坚持。
- 自主推进:阶段结果出来不停等;想问的问题改成四件套(问题/默认/理由/回滚)记台账;硬停止只在不可逆、方向性抉择、需用户独有信息时。
- 汇报:第一句结论;首次出现的代号/指标先定义;数字带读法(基线、样本量、显著性);用户说"说人话"就用任务本身讲。
- Git:直接提交 main、推送后才算落地;注释前缀 `# note (luojiaxuan):`;不留对话/历史痕迹;不写无根据的防御性代码;不加 AI 署名。
- 文档用中文;时间 PT;台账 `experiment-audit` 风格(结构化、可 grep)。
