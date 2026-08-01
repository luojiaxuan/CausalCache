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

附带一个反直觉的实测:`exec >>不可写路径` **不会**让脚本退出(照常执行后续行、
`exit=0`),只是重定向失败。配合 `>/dev/null` 启动就是全程静默、无迹可查。日志路径
必须先验证可写——`/data02/jaxan` 对 `sglang-omni` 不可写,容器内 `/data` 才行。

第四条教训——**多副本必须核对流量,别只看进程起没起来**。B 扫描 v1 在 GPU4/5 各起
一个服务器,但 runner 的 `--policy-endpoint` 是单值参数,18 个 env 全打 58900:
实测 GPU4 99% 利用率 / p0.log 523 条请求,GPU5 0% / p1.log **0 条**,一半算力空转
了整个配置。`nvidia-smi` 与逐副本请求计数是判断副本是否真在干活的最小证据,
"两个 server 都 /health 200" 完全不能说明问题。v2 改为每个配置并行跑两个 runner,
各带一半任务(subset-a/b,奇偶交错)与一半模拟器(fleet-a/b,9+9 不重叠)。
