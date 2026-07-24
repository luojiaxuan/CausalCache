# Frozen GUI-Owl 主表行:视觉记忆预算的零样本闭环表现(AndroidWorld 全 116 模板)

## 口径(用户 2026-07-24 冻结)
- roster:116 模板 × 3 instance(task_index 0/1/2)= 348 实例,全方法同一批固定实例;
- 主指标:**template 内先平均 3 instance,再对 116 template 等权 macro**;
- CI:bootstrap 以 **template 为 cluster**(不把 348 局当独立单位);
- Success = (1/116) Σ_t (y_t0+y_t1+y_t2)/3。

## 结果(1,740 局,116/116 模板满 3 instance)

| Memory budget | macro 成功率 | 95% CI(template-cluster) |
|---|---:|---|
| B=0(仅摘要) | 17.5% | [12.1, 23.6] |
| B=1 | 22.7% | [16.4, 29.3] |
| B=2 | 23.6% | [17.2, 30.2] |
| B=4 | 26.7% | [20.1, 33.6] |
| B=8 | 27.6% | [21.0, 34.2] |
| **Avg B>0** | **25.1%** | |

**B8 − B0 = +10.1pt,且单调递增。**

## 机制诊断(为什么 frozen 也吃记忆红利)

分臂统计(局级,n≈223/臂,中途快照):

| 臂 | 步数耗尽 | 主动终止 | 解析率 |
|---|---:|---:|---:|
| B0 | 104 | 63 | 0.984 |
| B8 | 59 | 107 | 0.982 |

增益**全部来自打破循环 / 恢复收尾能力**(step_budget_exhausted 104→59,policy_terminated 63→107),
解析率纹丝不动 → 不是格式效应。严格配对(n=221):B0 成功 49、B8 成功 77,B8 赢 32 输 4
(其中 21 局 B0 死于步数耗尽,7 局死于环境故障)。

边际结构:B0→B1 **+6.8pt(最大跳)**,B1→B2 −0.1,B2→B4 +4.6,B4→B8 +1.5——
一半以上增益来自"有那么一张最近的图",随后迅速饱和。

## 与论文立足点的关系(重要修正)

旧表述"GUI-Owl 只能看一张图"**站不住**,须改写为:

> frozen GUI-Owl 能从近期帧获得**浅层**收益(防打转、知收尾),但对历史内容的辨别力极弱
> (teacher-forced correct−shuffled = −0.000±0.003),因此剂量曲线迅速饱和;
> **如何选择历史**才是剩余收益空间。

对照实验 `oldest_B8`(同样 8 张图但取最老的)见 data/results/oldest_b8_control_v1/。

## 数据
hyper00 `/data02/jaxan/runs/frozen-main-row/`(1,740 局逐局 json),PENDING_HF_UPLOAD。
roster:`data/manifests/androidworld_full_suite_plan_v2.json`(348 实例,per-split seed 已修正)。
