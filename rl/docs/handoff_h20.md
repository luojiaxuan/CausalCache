# H20 交接 Runbook(给接手同学;完全独立账号/机器,从零起环境)

> 方法读 `rl_recipe.md`(自足,不需要读任何历史文档);本文只管
> "在你自己的机器上把它跑起来"。我方 hyper00 的 3×H200 smoke/mini 已
> 验证全链;你的环境要做的适配点在 §5,如实标注了哪些是**未验证**的。

## 0. 访问与保密(先办)

- **仓库不公开**。CausalCache GitHub 私有仓 + HF 私有 repo 由 luojiaxuan
  邀请你的账号为 collaborator(read 即可);**不要要求转 public**。
- 你需要的秘密只有:GitHub 访问、HF token(拉 executor 权重)、
  以及(仅当跑 ask_user 任务时)一个 OpenAI 兼容 key——GUI-only 训练
  不需要。
- 项目保密纪律是否延伸到你的共享机(容器/镜像/进程命名中性化),
  与 luojiaxuan 约定;我方侧的做法见 `rl/cua/README.md` 匿名节,
  全套机制(命名补丁、pyshim)可原样复用。

## 1. 物料清单

| 物料 | 来源 | 固定版本 |
|---|---|---|
| CUA-Lite | github.com/cua-lite/cua-lite(公开) | commit `2602509` |
| slime 子模块 | github.com/cua-lite/slime(公开) | branch v0.3.0(SSH url 需改 https) |
| **上游四补丁** | 本仓 `rl/cua/patches/cua_lite_2602509_worktree.patch`(`git apply` 即可) | 必打;内容与动机见 patches/ 下同名说明文件 |
| recipe 包 | 本仓 `rl/cua/`(`sglang_omni_rl` 包 + configs + fixtures + tests) | 以 main 最新为准 |
| executor 权重 | GUI-Owl-1.5-8B-Instruct(HF;若源 repo gated/不可得,由 luojiaxuan 的私有 HF 转交) | 对表所用快照 |
| 训练容器镜像 | slimerl/slime:v0.3.0(Docker Hub 公开) | v0.3.0 |
| env 基座镜像 | ghcr.io/tongyi-mai/mobile_world(公开 GHCR) | digest `b680380e…`(Dockerfile 已钉) |

## 2. 环境自举(每台 env 宿主一次)

1. 硬件门槛:`/dev/kvm` 可读写(MobileWorld emulator 必需)、CPU 按
   2.5 核/emulator 预算、每 env 容器内存上限 32g;
2. clone cua-lite@2602509 → `git apply` 四补丁 → `git submodule update
   --init`(slime)→ `uv sync --extra gym`(**注意 extra,漏了 env-server
   起不来**);
3. 把 `rl/cua/` 放在 cua-lite 同级(目录名随意,包经 symlink 进
   CUA_LITE_ROOT:`ln -s ../<recipe>/sglang_omni_rl cua-lite/sglang_omni_rl`);
4. `bash lite/gym/envs/mobileworld/scripts/install.sh` 构建 env 镜像
   (若沿用匿名纪律,构建后按 README 重打中性 tag——注意坑账:
   **改镜像名要同步检查按镜像名做发现的消费方**);
5. 冒烟:跑 `rl/cua/tests/test_parity_gui_owl.py`(204 例应全绿)→
   `scripts/e2e_probe.py` 单 episode(需一个临时 vLLM/sglang 端点)。

## 3. 服务拓扑(与我方 smoke 相同形态)

```
env 宿主:serve_env.py(:30100)+ selector service/trainer 侧车(CPU 即可)
训练容器(slimerl/slime:v0.3.0 起):run_grpo.sh
  → Ray + Megatron(train)+ sglang(rollout, 权重热换)
  → 经 CUA_LITE_ENV_SERVER_URL 打 env 宿主
```

发射参数模板与全部踩坑修正见 `rl/cua/scripts/run_mw_grpo.sh`
(动态选卡段按你的机器改)。**三个必设**,漏一个就是我们踩过的事故:

- 训练容器 `--ulimit nofile=524288:524288`(Ray 按核数 prestart worker,
  默认 1024 会把 raylet 的 FD 顶满,worker 首连超时且报错極具误导性);
- NVLS 不支持的 fabric 上设 `HAS_NVLINK=0`(补丁让预设生效);
- `CC_RET_DIR.path` 哨兵写**容器视角**路径。

## 4. 训练发射(H20 目标形态)

数据:`uv run python rl/cua/scripts/export_mw_tasks.py --split-json
rl/cua/fixtures/mw_split_v1.json --out-dir <dir>`(冻结切分,勿重切)。

单节点 8×H20 参考(**内存形态见 §5,TP 要调**):

```bash
ASYNC=1 NUM_TRAIN_GPUS=6 NUM_ROLLOUT_GPUS=2 TP_SIZE=<见§5> \
MODEL_ID=Qwen/Qwen3-VL-8B-Instruct HF_CKPT=<GUI-Owl 本地路径> \
ENV_ID=mobileworld PROMPT_DATA=<dir>/train.parquet \
ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=8 NUM_STEPS_PER_ROLLOUT=1 \
NUM_ROLLOUT=100 ENV_CONCURRENCY=64 \
ROLLOUT_MODULE=sglang_omni_rl.rollout_grpo \
CONFIG_PATH=<recipe>/configs/gui_owl/mobileworld_learned.yaml \
OPTIM_CPU_OFFLOAD=1 bash scripts/train/run_grpo.sh
```

selector 侧车照 `run_mw_grpo.sh` §3 原样起;监控清单与验收四条见
`rl_recipe.md` §6。

## 5. H20 适配点(如实:以下未在我方环境验证)

1. **显存形态**:H20 96GB vs 我方 H200 141GB。我方 TP2+optimizer CPU
   offload 时训练 rank 峰值 ~90GB/卡——**H20 上 TP2 大概率不够**,
   从 TP4 起步(8B 全参,32 卡预算下完全放得下),或
   `MAX_TOKENS_PER_GPU` 开 THD 动态打包(qwen3_vl 支持);
2. **多节点**:`run_grpo.sh` 写死 `--actor-num-nodes 1`。32×H20 若是
   4×8 节点,需要 Ray 多节点集群 + 该参数放开——slime 本身支持,
   但**我们没验证过**,预留一天调试;先单节点 8 卡把管线点通再扩;
3. **rollout 引擎数**:GPUS_PER_ENGINE=1 时 sglang 引擎数=rollout 卡数,
   `ENV_CONCURRENCY` 与 env 宿主的 emulator 数、CPU 核数联动
   (rollout-bound,wait_ratio 实测 ~0.7,并发是第一杠杆);
4. env 宿主可以与训练节点分离(env-server 就是为此设计),emulator
   吃 CPU 不吃 GPU——若 H20 节点 CPU 少,单独找 CPU 机器承载 env 池。

## 6. 已知事故速查(遇错先查这里)

八起 smoke 事故的死状→根因→修法全表在
`cua_lite_integration_20260825.md` §4.6;运维坑账(满盘假网络错、
镜像改名反噬发现机制、pkill 自匹配、目录迁移断 venv)在同文档与
`rl/cua/README.md`。selector 数学的唯一已知陷阱(PL slate_logprob 的
mask 必须逐步 clone)已修在包里,别回退。
