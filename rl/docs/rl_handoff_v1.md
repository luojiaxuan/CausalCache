# CausalCache-RL 交接包(2026-08-08)

给接手人的一份自足文档:跑起来需要什么、怎么跑、已知会踩什么坑、
以及**这个负载到底能不能吃下大规模 GPU**(先读第 0 节,它可能改变整个方案)。

---

## 0. 先回答一个问题:你的机器能不能跑这个负载?

**这不是一个 GPU-bound 负载。** 它由两半组成,瓶颈在前一半:

| 阶段 | 吃什么 | 现状实测 |
|---|---|---|
| **rollout(墙钟 ~70%)** | **每条 episode 一台 QEMU 虚拟机**(OSWorld Ubuntu),外加一个 8B 策略服务 | 8 任务 × G=6 = 48 episode ≈ 35-67 分钟 |
| train(墙钟 ~30%) | 8B 前向 + LoRA 反传,3 卡 DDP | ~8-13 分钟 |

因此有**三条硬前置**,任何一条不满足,机器再多也跑不了 rollout:

1. **`/dev/kvm` 可用**(嵌套虚拟化开启)。OSWorld 用 QEMU 起真实 Ubuntu 桌面;
   没有 KVM 只能软件模拟,慢到不可用。
2. **容器内可访问 docker daemon**(worker 在宿主侧起 VM 容器)。
3. **每台 VM 约 4-6GB RAM + 2-4 核**;48 路并发需约 250GB RAM / 100+ 核。

> **实测反例:tilde 的 Slurm 计算节点两条都不满足**(无 `/dev/kvm`、docker
> socket 拒绝),OSWorld 在那上面完全跑不起来。**如果这 128 台 H100 是同类
> 受管集群,接手人只能跑 train 那一段——而它不是瓶颈,收益很小。**
> 交接前请先让对方在一台节点上跑第 6 节的「五分钟环境自检」。

**如果三条都满足**,128×H100 的正确用法**不是**把当前配置铺开,而是改变
实验规模:当前每迭代只抽 8/44 个任务、G=6,受限于单机能开的 VM 数。有了大
集群应当:每迭代跑**全部 44 个任务 × G=8**(352 episode),每迭代的可用组从
3-6 提到 25-35,梯度质量与优化步数一起上一个量级。届时 GRPO 的同步结构
(rollout → train → 下一迭代)仍在,单迭代墙钟约等于**最慢一条 episode**,
所以横向扩展的收益是"每迭代数据量",不是"迭代更快"。

### ⚠️ 横向扩展的第一道硬墙:OSWorld 的全局端口锁(必须先改)

`desktop_env/providers/docker/provider.py` 用**一把全局文件锁**
(`/tmp/docker_port_allocation.lck`)罩住 **端口分配 + 整个容器启动**
(`containers.run(...)`),默认 `LOCK_TIMEOUT = 10` 秒。后果:

- **所有 VM 的启动被全局串行化**,每台约 5 秒;
- 并发一高,排队深度超过 10 秒的 worker **直接失败并丢掉整条 episode**。

我们的实测:8 路并发基本无事;**升到 16 路时 48 条里丢了 15 条**,失败计数
与 `Timeout: The file lock '/tmp/docker_port_allocation...'` 精确对应,
而且**不会触发 server 侧修复遍**(server 是好的),很容易被误读成"任务变难了"。

**两处必改(接手人在大规模跑之前先做):**

1. `LOCK_TIMEOUT` 10 → 600(让它排队等,而不是超时失败)。这是我们当前
   在 hyper00 上打的本地补丁,**不在 Git 里**(OSWorld 是第三方 checkout),
   接手人需自行施加并记录;
2. 数百路并发时,**光加超时不够**——串行启动本身成为吞吐上限
   (N 台 VM 至少需要 N×5 秒才能全部起来)。真正的扩展需要:
   把锁的粒度缩到只罩端口分配(容器启动移出临界区),或按主机/命名空间
   分片锁文件(如 `TMPDIR` 隔离 + 端口段分配),使不同分片互不争锁。
   **这是 128 卡规模下必须先解决的工程项,不解决的话卡再多也起不来 VM。**

---

## 1. 三十秒看懂这是什么

冻结 GUI-Owl-1.5-8B(一个参数不动),只训两个小件:

- **selector 打分头**(Linear(4096,1),4097 参数):决定把哪 B=2 张历史截图
  放进 prompt;
- **HGKV LoRA**(last-8 层 k_proj/v_proj,rank 8,655,360 参数):决定选进来的
  历史图 token 的 K/V 怎么被读。

奖励 = OSWorld 官方评测器的任务成败 0/1,无 shaping、无 reward model。
用 GRPO:同一任务采 G 条 rollout,组内相对优势。探索**唯一来源**是选帧的
Plackett-Luce 采样(τ=1 训练、τ=0 部署)。

方法细节读 [`rl_method.md`](rl_method.md),预注册契约读
[`rl_pivot_contract.md`](rl_pivot_contract.md)。

## 2. 当前状态(接手人必须知道)

- **v1 campaign 已结束,是配置失败而非假设失败**:20 迭代 + 双轮 held-out
  评测,RL 81/240 vs recent-B 88/240(Δ=−2.9pp,p=0.345,CI 跨零)。复盘
  发现**两条可学通道都没被训动**(selector 位移 4.4%、LoRA `lora_b` 范数
  0.04),根因是 20 迭代总共只有 67 次 optimizer.step() + lr 过小。
- **v2 campaign 正在跑**(2026-08-08 起,hyper00):selector lr 1e-3、
  LoRA lr 1e-4、grad_accum 4,其余不变。**前 3 迭代有硬门槛**
  (`selector_drift` ≥0.02),不达标立即停机重调。
- 证据、诊断与全部踩坑见 [`../README.md`](../README.md)。

## 3. 代码(Git 是 canonical)

- 仓库 <https://github.com/luojiaxuan/CausalCache>,分支 `main`。
  **需要确认接手人已获授权访问**(私有仓库)。
- `rl/` 目录**设计为可整体迁出**,只通过 4 处 import 依赖主仓
  (清单见 [`../README.md`](../README.md) 的「依赖边界」)。运行时
  `PYTHONPATH=<主仓>/code:<主仓>/rl/code`。
- 关键文件:

| 文件 | 作用 |
|---|---|
| `rl/runscripts/rl_iter_loop.sh` | 迭代编排(rollout→collect→train→server 重启),阶段 marker 断点续跑 |
| `rl/runscripts/rl_heldout_eval.sh` | held-out 双臂评测(RL vs 冻结+recent-B) |
| `rl/code/scripts/serve_osworld_rl_policy.py` | RL 专属策略服务(PL 采样 + 审计 + hidden selector) |
| `rl/code/scripts/collect_rl_trajectories.py` | 三键 join 成 GRPO 训练组 |
| `rl/code/scripts/train_causalcache_rl_grpo.py` | GRPO trainer(手工 all-reduce DDP) |
| `rl/code/scripts/make_hidden_head.py` | iter-0 selector 头 bootstrap |
| `rl/code/scripts/rl_eval_summary.py` | 双轮合并分析(完备性闸门 + 配对检验 + 噪声标定) |
| `rl/code/scripts/rl_task_probe.py` | reward hacking 探针 |
| `rl/code/scripts/rl_selector_drift.py` | 「参数动没动」诊断 |
| `rl/code/scripts/rl_selector_lr_sweep.py` | 离线 lr/熵 反事实扫描 |
| `rl/code/scripts/rl_noise_diagnostic.py` | 选帧—成败耦合诊断 |

## 4. 环境(逐字复制,版本是硬约束)

| 件 | 值 |
|---|---|
| 容器镜像 | `jaxanluo/sglang-omni:dev`(Docker Hub) |
| Python / torch | 3.12.3 / 2.11.0+cu130 |
| **transformers** | **必须 `5.6.0`** |
| VM 镜像 | `happysixd/osworld-docker:latest`(359MB) |
| OSWorld 仓库 | <https://github.com/xlang-ai/OSWorld> @ `0514b9a2` |
| VM 磁盘 | `Ubuntu.qcow2`,24,460,197,888 字节(约 23GB),OSWorld 首次运行自动下载 |
| 容器启动 | `--gpus all --shm-size=64g`,挂载持久盘,设 `HF_HOME`/`XDG_CACHE_HOME`/`PIP_CACHE_DIR`/`TMPDIR` 到持久盘 |

> **transformers 版本是 fail-closed 的**:冻结守卫会核对模型文件清单 +
> transformers 源码哈希,版本漂移直接拒绝启动。共享容器被别的项目升过包是
> 真实发生过的事故(h00 曾被升到 5.12.1,三个 server 全部拒启)。
> **迁移到任何"现成"容器,第一件事是核对这个版本。**

## 5. 数据与模型

| 件 | canonical 位置 | 体积 | 状态 |
|---|---|---|---|
| 策略模型 | `mPLUG/GUI-Owl-1.5-8B-Instruct` @ `06d5faecff74840bab2be2425e9c42667a5d04fc` | 17GB | HF,公开,**必须锁这个 revision** |
| 任务切分 train/held-out | `rl/data/manifests/osworld_rl_{train,heldout}_v1.json` | KB | Git canonical |
| 可学带 band v2(44 任务,10 域) | `rl/data/manifests/osworld_rl_train_band_v2.json` | 2KB | Git canonical(2026-08-08 入库) |
| iter-0 selector 头 | hyper00 `/data04/jaxan/rl2/iter-0/selector_bundle.pt` | 20KB | 可由 `make_hidden_head.py --seed 20260805` 确定性重建,**无需传输** |
| v1 campaign 轨迹与权重 | hyper00 `/data04/jaxan/rl/iter-{1..20}/` | ~10GB | 本地;intended `gavinlaw/causalcache-rl-osworld-trajectories`,`PENDING_HF_UPLOAD` |
| v1 held-out 评测原始结果 | hyper00 `/data04/jaxan/rl/eval-iter{12,20,20-r2}/` | ~3GB | 本地;汇总数已入 Git README,`PENDING_HF_UPLOAD` |

**跨机传输走 HF 私有仓库,不要过笔记本**(全局规则)。模型从 HF 直接拉;
VM 镜像让 OSWorld 自己下;真正需要传的只有 band v2 清单(KB 级)和
(可选)v1 轨迹用于复现诊断。

## 6. 五分钟环境自检(接手人先跑这个)

```bash
# 1) KVM
test -e /dev/kvm && echo "KVM OK" || echo "KVM 缺失 —— OSWorld 跑不了"
# 2) docker
docker info >/dev/null 2>&1 && echo "docker OK" || echo "docker 不可用"
# 3) 资源(48 路并发 VM 约需 250G RAM / 100+ 核)
nproc; free -g | awk '/Mem:/{print "RAM 可用 " $7 "G"}'
# 4) 冻结守卫版本
python3 -c "import transformers; assert transformers.__version__=='5.6.0', transformers.__version__; print('transformers OK')"
```

四项全绿才继续;第 1、2 项任一红灯,先回第 0 节重新评估方案。

## 7. 怎么跑

```bash
# 训练 20 迭代(v2 配置)
CTN=<容器名> RLH=<宿主迭代根> RLC=<容器内同一目录> \
REPO=<容器内仓库> WREPO=<宿主仓库> MODEL=<容器内模型目录> \
BAND=<容器内 band json> \
GPUS="2 3 4 5 6 7" SERVERS=6 TRAIN_GPUS="2 3 4" WORKERS=8 GEN_PARALLEL=2 \
SEL_LR=1e-3 LORA_LR=1e-4 GRAD_ACCUM=4 \
ITER_START=1 ITER_END=20 ./rl/runscripts/rl_iter_loop.sh
```

```bash
# held-out 双轮评测(两轮可并行,注意 PORT0/TAG 隔离)
ITER=20 GPUS="2 3 4" PORT0=19511 TAG=""   WORKERS_RL=6 WORKERS_RECENT=10 ./rl/runscripts/rl_heldout_eval.sh
ITER=20 GPUS="5 6 7" PORT0=19531 TAG="-r2" WORKERS_RL=6 WORKERS_RECENT=10 ./rl/runscripts/rl_heldout_eval.sh
```

```bash
# 合并读数(带完备性闸门,任一臂不足 120 条会拒绝出数)
python3 rl/code/scripts/rl_eval_summary.py --rounds <r1> <r2> --need 120
```

**每迭代必看两个数**(在 `iter_report.json`):`selector_drift`(参数动没动)
和 `optimizer_steps`(这一迭代更新了几次)。v1 的全部教训浓缩在这两个数上。

## 8. 已知的坑(全部真实踩过,按吃亏程度排序)

1. **参数没训动而不自知**:v1 白跑 20 迭代。→ 前 3 迭代盯 `selector_drift`。
2. **`mean_kl` 是每 episode 全 token 之和,不是每 token 均值**。0.2 看着像
   "信任域在工作",折算每 token 才 1e-3 nats,真实含义是"策略几乎没变"。
   **不要重蹈我这个误读。**
3. **worker 的 `--memory-arm` 合法值只有 `summary|full`**;recent-B 是
   server 侧默认行为(官方 serve 不带 selector 即 recent-B)。传 `recent`
   会让 8 个 worker argparse 秒退、整臂收 0 条。
4. **换臂/换配置时杀 server 必须等进程真正退出**:垂死进程仍应答
   `/health 200`,骗过健康检查,导致下一臂全灭。
5. **评测(50 步)比训练(30 步)吃显存得多**:索引遍最多 49 张缩略图
   ≈9.6K 视觉 token,注意力显存按 token 数平方涨,瞬时分配可达 18-36GB。
   **rl 臂 ≤2 worker/server**;recent 臂无索引遍,可放开到 3-4。
6. **长时间服务的 server 会显存爬升**:官方 serve 连续跑 ~2h 后开始小分配
   OOM。训练每迭代重启所以没暴露;长评测臂需依赖修复遍或中途重启。
7. **`groups.jsonl` 内嵌 episode 目录的容器视角绝对路径**,跨机不可复用,
   迁移后必须重跑 collect(秒级)。
8. **任一臂收 0 条必须 FATAL,不许盖章出数**;缺口集中在长尾困难任务,
   直接配对会系统性偏袒 RL 臂。
9. **worker 的 cwd 必须是 VM 镜像所在目录**,否则会重新下载 23GB 镜像。
10. **共享机上永远不要跑爆宿主 RAM**——sshd fork 不出来,全员失去登录。
11. **OSWorld 全局端口锁在高并发下静默丢 episode**(见第 0 节):16 路时
    48 条丢 15 条,server 侧完好因而**不触发修复遍**,极易被误读成任务变难。
    判据是 worker 日志里的 `Timeout: The file lock '/tmp/docker_port_allocation`
    计数——**每次提高并发后都要 grep 一次这个**。

## 9. 测量噪声与检出力(报结果时必须一起给)

- 同一冻结基线跨评测重测:39/120 vs 43/120,**逐任务翻转 11.7%**;
- 双轮内同臂跨轮翻转:recent 6.7% / rl 10.8%;
- 因此总数噪声约 **±4 个任务**;单轮 120 任务需**净胜约 11 个任务(≈9pp)**
  才达 p<0.05。**小于这个量级的差异,本设计看不见。**

只报点估计而不报这两个数,等于把噪声当信号——v1 的 "+6" 就是这么被我
一度当作正向证据的。
