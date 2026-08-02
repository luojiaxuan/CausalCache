# runscripts:主机侧运行脚本存档(可复用/可改)

按用户 2026-07-30 指示:每次跑实验的主机脚本入 Git,不再只存散在 /tmp。
命名 = 当时的实际文件名;路径硬编码对应各主机容器布局(h00=/data,
h01=/bigdata,详见 docs/run-mobileworld-review-ablations.md)。

- OSWorld 闭环链:osworld_serve_loop_h00.sh(serve_osworld_official_policy,
  adapter/selector 开关)+ osworld_worker_loop_h00.sh
  (run_osworld_benchmark_worker,--max-steps 默认 15,30 步诊断即改此参)。
- MobileWorld 32B 两臂:launch_32b_h00/h01.sh;8B selector 闭环:
  launch_frozensel_ft.sh / launch_selv5.sh;翻臂:flip_h00/h01.sh。
- 训练/评分:train_v5a.sh、train_32b_hgkv.sh、devscore_*.sh、
  train_frozen_selector.sh、frozen_singletons_h00.sh、fsets_hyper00.sh。
- 收割:harvest_arm.sh(双机 task→score 去重)。

- B 预算扫描(0/1/2/4/6/8,36 任务固定子集):bsweep_driver2.sh(驱动 v2)+
  split_bsweep.py(拆分子集与 fleet)+ bsweep_handoff.sh(v1→v2 在线交接)。
- 跨臂监控:watch_all_arms.sh(OSWorld 三臂完成数 + B 扫描推进,含停摆告警)。

教训三条(此目录脚本已体现):宿主 /tmp ≠ 容器 /tmp;docker exec 内
nohup 不保活(用 docker exec -d);监督器发射后不可动,server 无状态可随时迁卡。

第五条教训——**宿主与容器是两个执行域,跨域操作会静默半成功**。B 扫描的 v1→v2
在线交接脚本以 `sglang-omni` 身份跑在宿主上,而驱动和 runner 是**容器内的 root 进程**:

- `pkill` 杀 root 进程只拿到 EPERM,**静默失败**,旧驱动毫发无伤;
- 脚本没有 `set -e`,继续执行下一步清理,而清理走的是 `docker exec`(容器内有 root)
  ——于是**目录被真删了,驱动却没换掉**,落到最坏的中间态;
- 最后在宿主上启动新驱动,但脚本里全是容器路径,瞬间失败。

三条推论:跨域脚本里**每一步都要走同一个执行域**;杀进程后必须**回读确认**再做任何
破坏性操作;`pkill -f xxx` 的模式要写成 `[x]xx`,否则会匹配到承载它的那条命令行而自杀。

**这个坑同一天内犯了三次**,值得单独记住三种伪装:

1. 交接脚本用宿主 pkill 杀容器内驱动 → EPERM 静默失败,但同脚本的 `docker exec` 清理
   有 root 权限,造成"目录删了、驱动还在跑"。
2. 新写的补跑脚本把端点写死 `127.0.0.1` → h00 是 host 网络能跑通,h01 是 bridge 必须用
   容器 IP,服务器明明 READY 而 health 恒为 000。**同一份脚本在两台机器上语义不同。**
3. "暂停 B 扫描"时宿主 pkill 驱动失败(又是 EPERM),只杀掉了 runner 和服务器,
   驱动自己重启服务器继续跑了两个配置——**我以为暂停了,其实一直在跑**。

统一做法:凡是操作容器内进程,一律 `docker exec <容器> bash -lc 'pkill ...'`;
凡是拼接端点,先读 `docker inspect --format '{{.HostConfig.NetworkMode}}'` 决定用
`127.0.0.1` 还是容器 IP;**每次杀完都回读进程数确认为 0**。

第六条教训——**pgrep/pkill 的模式要用公共前缀,不能写完整文件名**。同一天犯了三次:

| 实际进程 | 用过的错误模式 | 为何不匹配 |
|---|---|---|
| `bsweep_driver2.sh` | `bsweep_driver.sh` | 正则 `.` 只吃一个字符,`2.sh` 是三个 |
| `osw_arm_r2.sh` | `osw_arm.sh` | 同上 |
| `serve_s100.sh`(承自 `osworld_serve_sel.sh`) | 只杀了新名字 | 旧名字的监督循环还在,几秒后原样复活 |

后果分两种,都很隐蔽:**监控侧**误报"驱动消失"(其实在跑),**清理侧**漏杀
(以为停了其实没停,B 扫描就因此在我"暂停"后又跑了两个配置)。

做法:模式取到公共前缀为止(`bsweep_driver`、`osw_arm`),**并且杀完回读计数**。
写变体脚本时(v2、r2)尤其要回头检查所有引用它的监控与清理逻辑。

第七条教训——**计数要数"完成物",不要数"容器物"**。同样犯了三次:

| 数了什么 | 后果 |
|---|---|
| `result.txt` 文件数 | 跨机重复被算两次,117 个文件里只有 102 个唯一任务,却报"117/110 完成" |
| 任务**目录**数 | 目录在任务开始时就建,把"在跑"当成"已完成",双臂 162/361 被报成 361/361 |
| 唯一 `task_id`(正确) | — |

判据:**完成物是 `result.json`,且必须按 task_id 去重**。任何"看起来提前完成"的
信号先当成计数错误来排查,不要先高兴。

附带一个反直觉的实测:`exec >>不可写路径` **不会**让脚本退出(照常执行后续行、
`exit=0`),只是重定向失败。配合 `>/dev/null` 启动就是全程静默、无迹可查。日志路径
必须先验证可写——`/data02/jaxan` 对 `sglang-omni` 不可写,容器内 `/data` 才行。

第四条教训——**多副本必须核对流量,别只看进程起没起来**。B 扫描 v1 在 GPU4/5 各起
一个服务器,但 runner 的 `--policy-endpoint` 是单值参数,18 个 env 全打 58900:
实测 GPU4 99% 利用率 / p0.log 523 条请求,GPU5 0% / p1.log **0 条**,一半算力空转
了整个配置。`nvidia-smi` 与逐副本请求计数是判断副本是否真在干活的最小证据,
"两个 server 都 /health 200" 完全不能说明问题。v2 改为每个配置并行跑两个 runner,
各带一半任务(subset-a/b,奇偶交错)与一半模拟器(fleet-a/b,9+9 不重叠)。

第八条教训——**容器的 GPU 绑定在创建时固定,`--gpus "device=0,1,2,3"` 会永久锁死可见范围**。
h01 的 `sglang-omni-jaxan` 就是这样建的:宿主 8 卡,容器只见 4 张,而其中 3 张被他人占用,
实际只剩 1 张可用。症状是 `torch.AcceleratorError: CUDA error: invalid device ordinal`
——`CUDA_VISIBLE_DEVICES=0,1,4,5` 在容器里指向不存在的序号。

判据:`docker inspect <c> --format '{{json .HostConfig.DeviceRequests}}'`,
若 `DeviceIDs` 不是空(空=all)就是被锁死了。建容器一律 `--gpus all`,
用哪几张由 `CUDA_VISIBLE_DEVICES` 在运行时决定。

**顺带**:重建容器还解决了另一个隐蔽问题。PID 1 是 `sleep infinity`,**不回收子进程**,
崩溃的训练留下大量 `<defunct>` 僵尸,它们**挂着显存不释放**——h01 的 GPU0 被误认为
"别人占了 125GB",实际是我自己崩溃 run 的泄漏,重建后归零。

## Tilde(Slurm 集群)接入要点

2026-08-02 首次接入。**`tilde` 别名是登录节点 `login-0`,没有 GPU**,
不要在上面跑任何训练/打分。资源:`main` 分区 36 节点 × 8×H100-80GB;
账户 `guests/zhen`,配额 `gres/gpu=8`(= 一个节点);文件系统单挂载 196 T、
已用 99%(3 T 可用)。

| 事项 | 做法 |
|---|---|
| 提交 | `srun`/`sbatch`,秒级拿到分配;**只信真正被 grant 的分配,不信 `sinfo` 的 idle 快照** |
| 容器 | `enroot 3.5.0`。`jaxanluo/sglang-omni:dev` 是**私有**镜像匿名拉不下来,用 **`hongccc/sglang-omni:dev`** |
| 数据 | 与 hyper 主机**互不可达**,但两边都能到 huggingface.co → 一律走 HF 私有仓库中转,不经本机 |
| 模型 | 登录节点能直连 HF,17 G 的 GUI-Owl 直接在 tilde 上拉,不要从 hyper 搬 |

三处输出噪音(都不是同一个来源,分别治):

- **交互登录横幅**来自服务端 `/etc/update-motd.d/{00-welcome,10-system-info,20-slurm-stats,30-ssh-users}`,
  **只在交互式登录时跑**——非交互 `ssh tilde 'cmd'` 本来就干净。远端 `touch ~/.hushlogin` 关掉。
- **客户端 INFO 噪音**:本机 `~/.ssh/config` 的 tilde 条目加 `LogLevel ERROR`。
- **`cpu-bind=MASK - worker-N ...`** 是 `srun` 自己打的,站点默认带 `verbose` 修饰符。
  `srun --quiet` **收不住**,要用 `SLURM_CPU_BIND=quiet`(或 `--cpu-bind=quiet,cores`)。

另一个会浪费时间的小坑:**tilde 的 `/tmp` 是会话隔离的**,`scp` 到 `/tmp/x`
之后另开一个 `ssh` 会话看不到那个文件。要么直投目标路径,要么走家目录中转。

第九条教训——**"只改一个 config 字段"不等于单变量对照,发射命令里的语料也是变量**。

v6 三臂(tightcap/midcap/nogain)本意是"相对 v4 只改 drift cap eps",三个 config 确实
逐字节核对过。但发射脚本是照抄 `train_gated36_long.sh` 改的,那个用
`--dataset-root /data/desktop-did-corpus-v5-selected`,我改成了 **corpus-v1**,
而 v4 基线用的是 **corpus-v4**:

| | dataset_root | dataset_manifest_sha256 |
|---|---|---|
| v4 基线 | `desktop-did-corpus-v4` | `167009bc…` |
| v6 首轮三臂 | `desktop-did-corpus-v1` | `e872828e…` |

**config 严格对齐、语料悄悄换掉,归因照样不成立**,三臂全部重跑(约 2h)。

判据要落在**产物**上而不是意图上:`run_manifest.json` 的 `cli_args` 与
`dataset_manifest_sha256` 与基线逐字段 diff,**唯一允许的差异是有意改的那几项**
(config 路径、output_root、frozen_score_cache)。这个 diff 十秒钟就能做,
应当成为每次"对照实验"训完的第一件事,而不是发现异常后才回头查。

**副产品:是下游的 schema 校验把这个错抓出来的。** corpus-v1 是
`causalcache.desktop_did_sample.v1`,config 声明 v2 —— 训练器宽容放过,
`score_sparse_history_arms.py` 拒绝执行。这类"上游宽容、下游严格"的不一致
是免费的错误探测器,**遇到它时应当先怀疑自己的输入,不要去放宽下游校验**。

重建前必查:`docker inspect --format '{{range .Mounts}}...'` 确认重要数据都在挂载上
(本项目是 `/data0X/jaxan → /bigdata|/data`),以及容器内除自己的作业外无他人进程。
重建命令须保留 `--ipc=host --shm-size=64g` 与 HF_HOME/XDG_CACHE_HOME/PIP_CACHE_DIR/TMPDIR
四个缓存环境变量,否则缓存会落回容器层。
