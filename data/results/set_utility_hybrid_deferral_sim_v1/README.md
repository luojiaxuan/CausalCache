# Hybrid deferral 策略 dev-set 模拟 v1

状态:`DEV_SET_SIMULATION_FOR_HYBRID_PREREGISTRATION_NOT_A_VERDICT`。

用 decision-v2 epoch-1 tune truth(content SHA `043d3e12...d4c86`)对**已冻结** scalar v2 checkpoint
(`77080c58...d873bfe`)模拟确定性 deferral 部署策略,不做任何重训。规则均为运行时可判定
(budget 与 `n_t` 已知),退回格子不需要模型 forward。

| 策略 | macro | delta vs recent | 95% CI | B1 | B2 | B3 | B4 | Long+ |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 纯 set_base(复现) | 0.4759 | +0.0240 | [+0.0041,+0.0442] | +0.072 | -0.007 | +0.007 | +0.024 | -0.012 |
| H2:B2→recent | 0.4776 | +0.0257 | [+0.0111,+0.0403] | +0.072 | 0 | +0.007 | +0.024 | -0.002 |
| **H1:B2 与 Long+→recent** | 0.4770 | **+0.0251** | **[+0.0111,+0.0393]** | +0.069 | 0 | +0.009 | +0.022 | 0 |
| H3:Long+→recent | 0.4764 | +0.0245 | [+0.0055,+0.0435] | +0.069 | -0.003 | +0.009 | +0.022 | 0 |

H1 在构造上**任何格子不劣于 recent**,macro 显著为正且 CI 下界高于纯 learned(0.0111 vs 0.0041)。

## 使用边界(必须遵守)

- deferral 格子是在观察 tune 数字后选择的:**本模拟只是混合部署合同的预注册输入**,不是任何 GO;
- 混合策略必须在触碰 untouched evaluation 前冻结为版本化合同;evaluation 与 closed-loop 是唯一裁判,
  失败即终止,不得回来重选格子;
- 不违反 learned general-`B` 路线的终止承诺:不训练 v4,不修改已死合同的 gate;本策略只是围绕
  已冻结 checkpoint 的新部署合同;
- 复现:[`summary.json`](summary.json)(含四种策略完整数字与 source binding)。
