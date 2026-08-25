# CausalCache × CUA-Lite recipe(out-of-tree,上游钉 2602509)

把 CausalCache 的 joint selector+executor RL(v2 recipe,见
`rl/docs/grpo_recipe_audit_20260824.md` 第三节)接进 CUA-Lite/slime。
上游仓库不 fork:本包经 `ROLLOUT_MODULE=sglang_omni_rl.rollout_grpo` 的
import 副作用注册 `gui_owl` family,rollout yaml 用 `agent_id: gui_owl`
绕过 factory。背景与决策日志:`rl/docs/cua_lite_integration_20260825.md`。

## 状态(2026-08-25)

- ✅ 判分通路:mobileworld env 终局 `/task/eval` 与官方 `scan_finished_tasks`
  同源;fid5 双路(截断/terminate)reward=float + 真实 eval_reason;
- ✅ gui_owl family:**204/204 字节级 parity**(vs 打补丁官方 agent;
  fixtures 见 `fixtures/parity_fixtures_v1.jsonl`,在 hyper00 官方 venv
  由 `scripts/gen_parity_fixtures.py` 生成);
- ✅ e2e 单 episode(hyper00,真环境+真 GUI-Owl-8B):8 步做完
  AcceptMeetingTask,**官方评测器判 reward=1.0**,CC_TRACE 窗口滑动正确;
- 🔴 进行中:slime 容器smoke(验收四条:reward 非全 0 / 选帧分布在变 /
  loss 有限 / 权重真热换)→ 3 GPU mini-run(train78 × G8 × 30-50 步)。

## 结构

```
sglang_omni_rl/
  gui_owl/            # model family:prompts(vendored 官方字节)/
                      # action_space(÷999 坐标、mobile_use 方言)/
                      # protocol(S 参数化官方布局)/ adapter / agent
  selector/           # PL 头(model)/ rollout 服务(service)/
                      # RLOO trainer(联合比率裁剪+归一化熵+drift 指标)
  registration.py     # import 副作用注册(agents 层无 out-of-tree env-var 钩子)
  rollout_grpo.py     # ROLLOUT_MODULE shim:slime 原版 convert + episode
                      # returns 旁路落盘(CC_RET_DIR,cc_episode 对账键)
configs/gui_owl/      # mobileworld_learned.yaml(主臂)/ _recent.yaml(对照)
fixtures/             # mw_split_v1.json(冻结切分 78/39 + strict 备选)
                      # parity_fixtures_v1.jsonl(306 step-case)
scripts/              # export_mw_tasks / gen_parity_fixtures / e2e_probe /
                      # run_mw_grpo.sh(发射草案)
tests/                # test_parity_gui_owl.py(204 case)
docs/                 # adapter_contract_notes.md(上游契约勘察)
```

## 关键设计与已记录偏差

- **ground truth = 打补丁的官方 agent**(产出了全部 P0.5/B2 数据;两臂
  内部一致)。与纯净官方的折叠格式差异(两空格/无 None 兜底/User response
  单列)只影响绝对数锚定;
- ask_user 文本格式偏差(CUA-Lite env note vs 官方 (Ask_user_response)
  前缀):GUI-only 117 任务训练不受影响,interaction 任务入训前需解决;
- `env_kwargs.extra_tools` 必须显式 `[terminate, response]`(上游默认 []
  会把 terminate noop 掉);
- selector 训练在侧车而非 slime 内核(Megatron 零手术;若需 per-step
  critic 再评估),executor 估计量用 slime `grpo`(组内基线+PPO 裁剪);
- 对账键:adapter 每 episode uuid → `sample.metadata.others.cc_episode`
  → slime Sample → shim 落盘 returns;selector service 决策日志同键。

## 复现命令(hyper00)

```bash
# parity(cua-lite checkout 根)
PYTHONPATH=/data01/jaxan/sglang-omni-rl/cc_recipe uv run python -m pytest \
  /data01/jaxan/sglang-omni-rl/cc_recipe/tests/test_parity_gui_owl.py -q

# e2e 单 episode(需 GUI-Owl vLLM 端点)
PYTHONPATH=/data01/jaxan/sglang-omni-rl/cc_recipe CC_HISTORY_N=3 CC_FRAME_POLICY=recent \
  uv run python /data01/jaxan/sglang-omni-rl/cc_recipe/scripts/e2e_probe.py \
  --task AcceptMeetingTask --llm http://127.0.0.1:41002/v1 --max-steps 8
```

## 下一步(smoke 发射清单)

1. slime 容器:pull `slimerl/slime:v0.3.0` → 重打白名单 tag(不覆盖既有)
   → 按 `scripts/train/slime/launch.sh` 的挂载/参数改规范名容器
   (sglang-omni-jaxan-N、--init、pyshim、高位端口);hyper00 checkout
   `git submodule update --init`(SSH url 需改 https);
2. `run_mw_grpo.sh` 首发逐段核验(动态选卡、env-server、侧车、容器内
   run_grpo.sh 参数——MODEL_ID=Qwen/Qwen3-VL-8B-Instruct(family 解析)
   + HF_CKPT=GUI-Owl 本地路径(权重));
3. 验证 cc_episode 端到端到达 returns 落盘(shim 有 fail-loud);
4. smoke 四条验收 → mini-run → 32×H20 runbook。

资源纪律:mw_rl 池 8 台(前一 session 保留)对 CUA-Lite 无用——它自管
容器;smoke 通过后按收尾清单删除。mw_random 池 + GPU6 vLLM 归并行
session 的探针线,勿动。
