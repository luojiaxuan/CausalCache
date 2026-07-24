# 交接文档:history-gated 主线执行状态(2026-07-24,Claude→GPT session)

Claude session 因 Fable 额度耗尽交接。本文档自足:读完即可接管全部在飞任务。
分支:`exp/history-gated-mainline-v1` 与 `main` 同步(HEAD 7351223)。协议文档
(必读,按此顺序):`docs/history_gated_mainline_v1.md`(V1 总契约)→
`docs/sealed_zero_shot_policy_matrix_v1.md`(含 V2 修订:全量 116×2 零样本)→
`docs/selector_v1_protocol.md`(含 V2 修订:两阶段 selector)。

## 一、已完成的里程碑(不需要重做)

1. **Adapter 正式 gate PASS,hg-s100 冻结**(sha256 前缀 `8f2cc49e1aa0`):
   全量 165 heldout、n=1,220 配对组,c−b0 +0.1372[+0.1296,+0.1449]、c−shuf
   +0.0016[+0.0003,+0.0030]、c−irrel +0.0108[+0.0078,+0.0139],B0 parity 逐位 0。
   详见 `data/results/hgkv_gate_v1/README.md`。弱点(wrong-history drift +0.07)已记录。
2. **闭环接线 + canary 12/12**:worker/runtime 支持 `--adapter-type history_gated_kv`。
3. **paper 决策(用户裁定)**:全线零样本;margin-SFT 从 paper 删除;主表 = 全量
   116 template × 2 实例(index 0/1)× {Frozen, Full-layer fl-s75, History-gated hg-s100}
   × {summary_B0, recent_B4, recent_B8} = 2,088 局;三视图报告(headline 全 116 /
   动作兼容子集剔 2 绘图+7 trivial-verify / 硬删仅 env-init 失败者)。
4. **selector 标签基建**:75,628 singleton 已用 hg-s100 重打(零重复零非有限);
   b0 冻结分 10,680 可复用;质量门 heldout 通过(oracle-over-Recent Δ+0.0205
   CI[+0.0187,+0.0223];STOP 弱:73.4% 全正——预算纪律是最大风险)。

## 二、在飞任务(接管重点)

### A. 全量矩阵(task #8,paper 主表)
- roster:`data/manifests/androidworld_full_suite_plan_v1.json`(232 实例,sha 1dd48941);
- **hyper00:6 worker 在跑**(容器 `*R8`-`*R13`,GPU 2-7,emulator 28300-28305,
  分配 707 cell);**hyper01:已按用户指令暂停**(其 1,054 cell 待跑,selector 优先);
- 已完成记录:两台 `/data02/jaxan/runs/sealed-matrix-v1/{frozen,full,gated}/*.json`
  (逐 cell 落盘,`ARM-TASK-IDX.json`;void/attempt 目录不算);
- **续跑方法**(幂等):重算缺失 = 全量 roster 减去两台已有 json(排除
  infrastructure_failure=True),用 `/tmp/gen_relaunch.py` 模式生成 worker(样板在两台
  `/tmp/gen_full.py` + `/tmp/relaunch.json`);**checkpoint 路径按主机**:
  hyper00 hg=`/work/runs/hgkv-formal-v1/lora-step100.pt` fl=`/work/runs/ody-margin-v2/lora-step75.pt`;
  hyper01 hg=`/work/runs/hgkv-eval/hg-s100.pt` fl=`/work/runs/hgkv-eval/fl-s75.pt`(同 sha);
- 聚合:配对 template 级差 + 10k bootstrap CI;void(infra=True 且 0 步)不进分母不占
  重试;三视图见协议 V2。

### B. Selector(tasks #9-11,用户当前最高优先)
- **conditional 渲染在跑**(hyper00 CPU,与任何 GPU 无关,会自己跑完):容器
  `sglang-omni-jaxan-07241303ch`(heldout)/`...ct`(train),输出
  `/data02/jaxan/artifacts/sft/ody-labels-cond/{heldout,train}/`,完成标志 `DONE` 文件
  (含 failed_shards 计数)。预计 heldout ~4.5-5.5 万行、train ~25-30 万行;
- **singleton 分数**(现成):hyper00 `/data02/jaxan/artifacts/sft/hgkv-selector-scores/`
  (singleton-train.jsonl 64,264 / singleton-heldout.jsonl 11,364 / b0-frozen-all.jsonl 10,680);
- **HGKV-readout 抽取器**(已提交,9/9 单测):`code/scripts/extract_hgkv_readout_features.py`,
  参数同打分器(--dataset-root 指 singleton 渲染,--lora-checkpoint hg-s100,rank8 alpha16,
  --shard-index/count 分片)。**下一步:hyper01 8 卡分片抽 75,628 个 singleton 特征**
  (数据 `/data02/jaxan/artifacts/sft/ody-labels-single`,heldout 段在
  `/data02/jaxan/artifacts/sft/ody-labels-held`);
- **cheap 基线**:`code/scripts/train_cheap_selector_v1.py` 已提交(agent 训练可能没跑完,
  产物应在 hyper00 `/data02/jaxan/runs/cheap-selector-v1/`,没有就自己跑,CPU 即可);
- **执行序**(协议 V2 §执行序):抽特征 → Stage-1 singleton scorer(轻头,预测 U_act)
  → conditional heldout 打分(渲染 DONE 后,scorer 现成支持 cond 样本,resume 键含
  restored_set_key)→ Stage-2 set-conditioned selector(1 selected-set attn + 1
  candidate-query attn + marginal/STOP 头,B≤4,singleton scorer 初始化)→
  **正式 selector gate:B1/B2/B4 对 selector/Recent/Similarity/Random 选中的完整集合
  重渲染重打分比真实 U_act(S),禁 ΣU 外推**;
- Stage-2 训练器注意:另一会话的 train_selector_v1.py 是自由形态 multimodal,不在预注册
  赛道;主方法必须是 HGKV-readout set-conditioned(协议 V2 §3)。

### C. 矩阵与 selector 的 GPU 协调
用户指令:selector 优先。hyper01 8 卡给 selector(特征抽取 + conditional 打分);
hyper00 6 卡继续矩阵。selector GPU 需求空档期可恢复 hyper01 矩阵 worker(续跑方法见 A)。

## 三、工程坑清单(全部踩过,勿重蹈)

1. **容器内路径是 `/work`**,不是宿主 `/data02/jaxan`(checkpoint/数据全同理);
2. **jaxanluo/sglang-omni:dev 有 CUDA banner 打进 stdout**:禁止用它 cat/管道传文件或
   捕获其 stdout 当数据;跨机传文件用 `ssh A cat | ssh B 'cat > f'` + 双端 sha256;
   容器内需要无 banner 时用 ubuntu:22.04(但它没有 python3);
3. **GPU 隔离**:发射 GPU 任务前必须 nvidia-smi 确认目标卡无在跑任务(标签打分曾团灭
   矩阵 worker);
4. **日志是 append 的**:判错只看当前轮(tail),grep 全文件会命中陈旧 Traceback;
   监视器用完成计数不用日志 grep;
5. **emulator**:容器内服务固定绑 5000,由 `-p 127.0.0.1:HIGH:5000` 映射(高位端口,
   `docker port <c> 5000/tcp` 查);服务死了进容器
   `cd / && nohup python3 -m server.android_server > /tmp/server.log 2>&1` 复活;
   **不要 kill 容器 1 号进程**;并行 boot 多台新 emulator 会因装 APK 冲突失败,
   现有 15 台(h00 28300-05,h01 28200-02,04,05,07,09,10,11)够用;
6. **sealed guard**:worker 需 `--allow-sealed-split` 才能载 split=test/full 的 plan;
   plan 必须是带 split 字段的清单(data/manifests/*_plan_*.json),不是模板 config;
7. **state-context 陷阱**:渲染器要的是
   `/data02/jaxan/runs/causalcache-variable-history-context-v2-1285ba8/state-context.jsonl`
   (sha 前缀 bb88bcf6);`9d9ef991` 那个 states.jsonl 缺 fits 字段会**静默产 0 样本**;
8. 渲染三输入:source
   `/data02/jaxan/artifacts/causalcache-set-utility-variable-history-source-v1-a7213db/trajectory-shards`、
   annotations `/data02/jaxan/source/guiodyssey-raw-annotations/annotations`、state-context 见上;
9. 容器命名 `sglang-omni-jaxan-<MMDDHHMM><后缀>`,用完即删,只清自己的;
10. 打分器 resume 键 = (pair_group, variant, singleton_event_step_id, restored_set_key),
    向后兼容;singleton 复用渲染的变体名是 single{N},候选 id 在
    memory_config.restored_event_step_ids[0]。

## 四、监控中的自动任务(Claude session 关闭后失效,需自查)

- 矩阵完成:数两台 sealed-matrix-v1 非 infra json 到 2,088;
- conditional 渲染完成:两个 DONE 文件;
- hyper00 6 个矩阵 worker 与 2 个渲染容器正常时自会退出,记得清尸体。

## 五、任务清单(接管后按序)

1. hyper01 8 卡抽 singleton HGKV-readout 特征(抽取器命令见 §二B);
2. cheap 基线补跑/验收(heldout 诊断表:Spearman/top-1/sign-precision/regret + B=1 真实
   U_act vs Recent CI);
3. Stage-1 singleton scorer 训练(特征→U_act 轻头);
4. conditional heldout 打分(渲染 DONE 后);
5. Stage-2 set-conditioned selector + 正式 selected-set gate(B1/B2/B4,CI 稳胜三基线);
6. 空档恢复 hyper01 矩阵 → 2,088 补齐 → 三视图主表 + 配对 CI;
7. 每步入库推送(main 与 exp 双分支保持同步;progress.md 冲突按时间并集解决)。

## 六、临交接前的最新落地(2026-07-24 13:1x)

- **conditional heldout 渲染完成**:45,227 行,failed_shards=0(train 段 ~10.9 万行仍在跑,
  等它的 DONE);→ 可立即用 hyper01 空卡打分(scorer 现成);
- **cheap 基线已训完并验收**(hyper00 `/data02/jaxan/runs/cheap-selector-v1/`,
  HistGradientBoostingRegressor,GroupKFold(5) by episode,21 特征,join 零缺失):
  heldout B=1 真实 U_act:model +0.1102 vs Recent-1 +0.1132 vs Random-1 +0.1034 vs
  oracle +0.1337;**配对 model−Recent = −0.0030 CI[−0.0046,−0.0014](认证为负,
  输给 Recent)**;model−Random +0.0068 CI 正。结论:平台通用元数据特征不足以胜
  Recency —— 这正是 HGKV-readout 主方法的动机论据,cheap 基线按协议入 appendix;
- HGKV-readout 抽取器已提交(7351223,9/9 单测),待 hyper01 8 卡执行(§二B)。
