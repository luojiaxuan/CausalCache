# CausalCache

**Restoration-Guided Memory for Long-Horizon GUI Action Prediction**

目标会议：AAAI。项目当前优先验证一件事：由 restoration utility 监督的 set predictor，能否在未参与训练的 GUI states 上超过 recent 与 OCR/RGB heuristic。

## 当前结论

- Restoration signal 存在，但旧版 conditional gate 与 independent gate 都没有在 untouched confirm 上稳定超过廉价 heuristic。
- GUIOdyssey 已固定 1,200 trajectories。旧 processor artifact 只物化每条轨迹的 anchor/terminal 两个 query，不能算 state-level 扩数完成。
- D2 已证明 strict-determinism runtime 可稳定复现先前的异常 state。
- 旧的 all-or-nothing throughput / stability 协议不再阻塞探索主线。新 MVP 按 state 接受：稳定 state 产标签，不稳定 state 记录并跳过。
- 旧 `scale-v1` 仅有 355 个 anchor states，现降级为 anchor-only pilot，不再作为扩数结果或继续调参依据。
- Oracle-independent `J` 在 B1/B2 恢复 exact utility 的 99.39%/90.06%，证明 independent restoration objective 在这批数据上有效；learned models 仍有明显 distillation gap。
- Dense per-step census 共 12,792 states（10,680 train / 1,066 tune / 1,046 evaluation）。现有 artifact 可复用 12,635 个；其余 157 个来自 20 条超长 trajectory 的中段截图缺口，需从 pinned raw source 补齐。

完整历史与失败记录保留在 [`docs/progress.md`](docs/progress.md)，但不应把历史 formal contract 当成当前 MVP 的执行清单。

## 方法

完整历史策略给出参考行为。将历史事件降为摘要后，恢复 coalition `S`，得到与完整历史行为的距离 `D(S)`：

\[
U(S)=D(\varnothing)-D(S).
\]

训练 predictor `U_θ(q,C,m_S)` 直接预测 subset utility。模型始终看到全部候选和 selected mask；budget 只进入 at-most-`B` search，不写死在模型中。

当前比较：

- pairwise-additive；
- DeepSets；
- Set Transformer；
- recent、OCR/RGB、oracle-independent `J`；
- exact subset oracle。

## 当前执行主线

- 同一 trajectory 的所有 eligible decision steps 保持在同一 split；
- 每个 state 使用最近 4 个 non-current events，生成全部 11 个 `|S|<=2` coalition labels；
- census 见 [`data/results/set_utility_dense_v1/census.json`](data/results/set_utility_dense_v1/census.json)；
- 先补齐 157 个 image gaps，再对 12,792 states 分片生成 labels、训练 predictor。

## Source of Truth

| 内容 | 位置 | 状态 |
|---|---|---|
| 代码、配置、论文、轻量结果 | 本 Git 仓库 `main` | canonical |
| Processor substrate | [HF dataset](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-new-development-mobile/tree/c20bab8df424dc9e45ece1084f3d1dc035dd1ed8/artifacts/processor-freeze-v2-image-contract-repair) | immutable，23 files / 18.73 GB |
| GUI-Owl snapshot | `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc` | frozen |
| Anchor-only pilot labels/features | [HF dataset@a95ce68b](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-new-development-mobile/tree/a95ce68bd628daaec40a7575847c9db584f20dc4/artifacts/set-utility-scale-v1-ac0ef27) | deprecated pilot；仅保留复现 |
| Anchor-only pilot checkpoints | [HF model@365f3882](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/365f3882658b40eccb64c9565ae8639986c59e82) | deprecated pilot；仅保留复现 |

大文件若暂时无法上传 HF，必须保存在个人 persistent storage，并在本 README 或结果文档记录精确路径与 `PENDING_HF_UPLOAD`。

## 仓库结构

- `code/`：package、scripts、configs、tests；
- `data/`：轻量 manifests、summaries、结果索引；
- `docs/`：当前实验说明和历史进展；
- `paper/`：AAAI LaTeX、references、figures/tables；
- `ablations/`：interaction 与 subset-search 分析设计。

## 本地验证

```bash
PYTHONPATH=code .venv/bin/pytest -q \
  code/tests/test_set_utility_dense.py \
  code/tests/test_set_utility_mvp.py \
  code/tests/test_set_utility_label_inputs.py

PYTHONPATH=code python3 -m compileall -q \
  code/causalcache \
  code/scripts/census_set_utility_dense_states.py \
  code/scripts/train_set_utility_mvp.py
```

## Go / No-Go

MVP 的首要判断不是 closed-loop，而是 held-out utility selection：

- 若 dense predictor 稳定超过 OCR/RGB，并明显缩小 exact-oracle gap：继续扩充 `n=8,16` 数据，再做 matched-NLL 与 closed-loop。
- 若 oracle-independent `J` 有效但 learned models 失败：改进表示和训练数据。
- 若 exact oracle 有 signal、但所有可学习目标和简单 baseline 持平：重新审视 predictor objective。
- 不因单个 near-tie / unstable state 让整个数据集归零；报告接受率和失败类别。

## Paper Claim 边界

“Causal”只指对冻结策略行为进行受控 restoration intervention 得到的 counterfactual attribution；不声称识别环境结构因果，也不把方法包装成 world model。
